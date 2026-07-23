use std::{future::Future, pin::Pin};

use rand::{rngs::StdRng, SeedableRng};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;
use truco_bot_core::{
    build_llm_request, parse_llm_response, resolve_weighted_action, turn_for_match, BotDecision,
    BotError, BotPlan, LlmBotRequest,
};
use truco_engine::{Action, Match, Player};

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

const OPENAI_CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";
const OPENAI_MODELS_URL: &str = "https://api.openai.com/v1/models";
const ANTHROPIC_MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_MODELS_URL: &str = "https://api.anthropic.com/v1/models";
const OPENROUTER_CHAT_COMPLETIONS_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
// Curated-but-fresh: only JSON-schema-capable models (so `response_format` works),
// ordered by the last week's real usage. The list self-refreshes without a hardcoded
// model allowlist; `extract_provider_models` guarantees major-lab family coverage,
// backfills by popularity, and caps the count.
const OPENROUTER_MODELS_URL: &str =
    "https://openrouter.ai/api/v1/models?supported_parameters=structured_outputs&sort=top-weekly";
const OPENROUTER_DEFAULT_TITLE: &str = "Baixada Truco";
// Popularity-ordered OpenRouter catalogs are capped so the picker stays a curated
// shortlist rather than the provider's full ~300-model list.
const OPENROUTER_MODEL_LIMIT: usize = 25;
// Two-tier curation: guarantee the major labs stay represented even when the
// weekly-popularity ranking is dominated by other models. Pass 1 seeds up to
// OPENROUTER_FAMILY_LIMIT models per family prefix (in popularity order);
// pass 2 backfills the remaining slots from the overall ranking.
const OPENROUTER_FAMILY_PREFIXES: &[&str] =
    &["openai/", "anthropic/", "google/", "deepseek/", "z-ai/"];
const OPENROUTER_FAMILY_LIMIT: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmProviderKind {
    #[serde(rename = "openai")]
    OpenAi,
    Anthropic,
    #[serde(rename = "openrouter")]
    OpenRouter,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmProviderModel {
    pub id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmProviderCatalog {
    pub provider: LlmProviderKind,
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    pub models: Vec<LlmProviderModel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderConfig {
    pub kind: LlmProviderKind,
    pub model: String,
    pub api_key: String,
    pub chat_url: String,
    pub models_url: String,
}

/// Per-kind environment contract: which vars name the key/model and what the
/// default endpoints are. Keeps `resolve` free of a big per-kind match.
struct ProviderEnvSpec {
    key_var: &'static str,
    model_var: &'static str,
    base_url_var: &'static str,
    base_url_default: &'static str,
    models_url_var: &'static str,
    models_url_default: &'static str,
}

impl LlmProviderKind {
    fn env_spec(self) -> ProviderEnvSpec {
        match self {
            Self::OpenAi => ProviderEnvSpec {
                key_var: "OPENAI_API_KEY",
                model_var: "TRUCO_OPENAI_MODEL",
                base_url_var: "TRUCO_OPENAI_BASE_URL",
                base_url_default: OPENAI_CHAT_COMPLETIONS_URL,
                models_url_var: "TRUCO_OPENAI_MODELS_URL",
                models_url_default: OPENAI_MODELS_URL,
            },
            Self::Anthropic => ProviderEnvSpec {
                key_var: "ANTHROPIC_API_KEY",
                model_var: "TRUCO_ANTHROPIC_MODEL",
                base_url_var: "TRUCO_ANTHROPIC_BASE_URL",
                base_url_default: ANTHROPIC_MESSAGES_URL,
                models_url_var: "TRUCO_ANTHROPIC_MODELS_URL",
                models_url_default: ANTHROPIC_MODELS_URL,
            },
            Self::OpenRouter => ProviderEnvSpec {
                key_var: "OPENROUTER_API_KEY",
                model_var: "TRUCO_OPENROUTER_MODEL",
                base_url_var: "TRUCO_OPENROUTER_BASE_URL",
                base_url_default: OPENROUTER_CHAT_COMPLETIONS_URL,
                models_url_var: "TRUCO_OPENROUTER_MODELS_URL",
                models_url_default: OPENROUTER_MODELS_URL,
            },
        }
    }
}

impl ProviderConfig {
    pub fn from_env(
        kind: LlmProviderKind,
        requested_model: Option<String>,
    ) -> Result<Self, LlmProviderError> {
        Self::resolve(kind, requested_model, None)
    }

    /// Like `from_env` but lets a caller supply the API key directly (bring your
    /// own key). A present, non-blank override wins over the environment; the
    /// model and endpoints still resolve from the request/environment as usual.
    pub fn from_env_with_key(
        kind: LlmProviderKind,
        requested_model: Option<String>,
        api_key_override: Option<String>,
    ) -> Result<Self, LlmProviderError> {
        Self::resolve(kind, requested_model, api_key_override)
    }

    fn resolve(
        kind: LlmProviderKind,
        requested_model: Option<String>,
        api_key_override: Option<String>,
    ) -> Result<Self, LlmProviderError> {
        let spec = kind.env_spec();

        let model = requested_model
            .filter(|value| !value.trim().is_empty())
            .or_else(|| std::env::var(spec.model_var).ok())
            .ok_or_else(|| {
                LlmProviderError::MissingConfig(format!(
                    "Missing bot model. Set bot_model in the request or {} in the environment.",
                    spec.model_var
                ))
            })?;

        let api_key = api_key_override
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                std::env::var(spec.key_var)
                    .ok()
                    .filter(|v| !v.trim().is_empty())
            })
            .ok_or_else(|| {
                LlmProviderError::MissingConfig(format!(
                    "Missing {} for the {} bot.",
                    spec.key_var,
                    provider_display_name(kind),
                ))
            })?;

        Ok(Self {
            kind,
            model,
            api_key,
            chat_url: std::env::var(spec.base_url_var)
                .unwrap_or_else(|_| spec.base_url_default.to_string()),
            models_url: std::env::var(spec.models_url_var)
                .unwrap_or_else(|_| spec.models_url_default.to_string()),
        })
    }
}

fn provider_display_name(kind: LlmProviderKind) -> &'static str {
    match kind {
        LlmProviderKind::OpenAi => "OpenAI",
        LlmProviderKind::Anthropic => "Anthropic",
        LlmProviderKind::OpenRouter => "OpenRouter",
    }
}

#[derive(Debug, Error)]
pub enum LlmProviderError {
    #[error("{0}")]
    MissingConfig(String),
    #[error("{0}")]
    Transport(String),
    /// The provider rejected the request for lack of credits/quota (HTTP 402) —
    /// e.g. a shared key that hit its spend cap. Kept distinct from `Transport`
    /// so the service can hand the UI a "shared key exhausted, bring your own"
    /// signal instead of a generic upstream failure.
    #[error("{0}")]
    Exhausted(String),
    #[error("{0}")]
    InvalidResponse(String),
    #[error("{0}")]
    Bot(#[from] BotError),
}

#[derive(Debug, Clone, Copy)]
pub enum HttpMethod {
    Get,
    Post,
}

#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub body: String,
}

pub trait LlmTransport: Clone + Send + Sync + 'static {
    fn send(&self, request: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, LlmProviderError>>;
}

#[derive(Debug, Clone, Default)]
pub struct ReqwestTransport {
    client: Client,
}

impl ReqwestTransport {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

impl LlmTransport for ReqwestTransport {
    fn send(&self, request: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, LlmProviderError>> {
        Box::pin(async move {
            let mut builder = match request.method {
                HttpMethod::Get => self.client.get(&request.url),
                HttpMethod::Post => self.client.post(&request.url),
            };

            for (name, value) in request.headers {
                builder = builder.header(&name, &value);
            }
            if let Some(body) = request.body {
                builder = builder.json(&body);
            }

            let response = builder
                .send()
                .await
                .map_err(|error| LlmProviderError::Transport(error.to_string()))?;
            let status = response.status();
            let body = response
                .text()
                .await
                .map_err(|error| LlmProviderError::Transport(error.to_string()))?;

            if !status.is_success() {
                let message = format!("LLM provider request failed with {status}: {body}");
                // 402 Payment Required is how OpenAI-compatible providers (incl.
                // OpenRouter) report an exhausted key/spend cap. Surface it
                // distinctly so the UI can offer bring-your-own-key.
                if status.as_u16() == 402 {
                    return Err(LlmProviderError::Exhausted(message));
                }
                return Err(LlmProviderError::Transport(message));
            }

            Ok(HttpResponse { body })
        })
    }
}

#[derive(Debug, Clone)]
pub struct LlmProviderBot<T = ReqwestTransport> {
    transport: T,
    config: ProviderConfig,
    rng: StdRng,
    last_decision: Option<BotDecision>,
}

impl LlmProviderBot<ReqwestTransport> {
    pub fn from_env(
        kind: LlmProviderKind,
        requested_model: Option<String>,
        seed: Option<u64>,
    ) -> Result<Self, LlmProviderError> {
        Self::from_config(
            ReqwestTransport::new(),
            ProviderConfig::from_env(kind, requested_model)?,
            seed,
        )
    }

    pub async fn fetch_catalogs() -> Vec<LlmProviderCatalog> {
        let transport = ReqwestTransport::new();
        let mut catalogs = Vec::new();
        for provider in [
            LlmProviderKind::OpenAi,
            LlmProviderKind::Anthropic,
            LlmProviderKind::OpenRouter,
        ] {
            catalogs.push(fetch_catalog(&transport, provider).await);
        }
        catalogs
    }

    /// Build a bot from a caller-supplied API key (bring your own key) instead
    /// of the environment's shared key. The key is used only to construct the
    /// request; it is never logged or persisted here.
    pub fn with_api_key(
        kind: LlmProviderKind,
        requested_model: Option<String>,
        seed: Option<u64>,
        api_key: String,
    ) -> Result<Self, LlmProviderError> {
        Self::from_config(
            ReqwestTransport::new(),
            ProviderConfig::from_env_with_key(kind, requested_model, Some(api_key))?,
            seed,
        )
    }
}

impl<T: LlmTransport> LlmProviderBot<T> {
    pub fn from_config(
        transport: T,
        config: ProviderConfig,
        seed: Option<u64>,
    ) -> Result<Self, LlmProviderError> {
        let derived = seed.unwrap_or_else(rand::random::<u64>) ^ 0x11A0_7000;
        let mut bytes = [0_u8; 32];
        bytes[..8].copy_from_slice(&derived.to_le_bytes());
        Ok(Self {
            transport,
            config,
            rng: StdRng::from_seed(bytes),
            last_decision: None,
        })
    }

    pub fn last_decision(&self) -> Option<&BotDecision> {
        self.last_decision.as_ref()
    }

    pub async fn choose_action(
        &mut self,
        game: &Match,
        player: Player,
    ) -> Result<Action, LlmProviderError> {
        Ok(self.choose_decision(game, player).await?.action)
    }

    pub async fn choose_decision(
        &mut self,
        game: &Match,
        player: Player,
    ) -> Result<BotDecision, LlmProviderError> {
        let turn = turn_for_match(game, player)?;
        let request = build_llm_request(&turn);
        let raw = self
            .transport
            .send(build_provider_request(&self.config, &request))
            .await?;
        let content = extract_provider_content(self.config.kind, &raw.body)?;
        let response = parse_llm_response(&content)
            .map_err(|error| LlmProviderError::InvalidResponse(error.to_string()))?;
        let plan = BotPlan {
            choices: response.choices,
            reasoning: if response.reasoning.is_empty() {
                None
            } else {
                Some(response.reasoning)
            },
        };
        let action = resolve_weighted_action(&plan, &turn.legal_actions, &mut self.rng)
            .map_err(LlmProviderError::from)?;
        let decision = BotDecision { action, plan };
        self.last_decision = Some(decision.clone());
        Ok(decision)
    }
}

async fn fetch_catalog<T: LlmTransport>(
    transport: &T,
    provider: LlmProviderKind,
) -> LlmProviderCatalog {
    let default_model = std::env::var(provider.env_spec().model_var).ok();

    let config = match ProviderConfig::from_env(provider, default_model.clone()) {
        Ok(config) => config,
        Err(LlmProviderError::MissingConfig(message)) => {
            return LlmProviderCatalog {
                provider,
                enabled: false,
                default_model,
                models: fallback_models_from_env(provider),
                note: Some(message),
            }
        }
        Err(error) => {
            return LlmProviderCatalog {
                provider,
                enabled: false,
                default_model,
                models: fallback_models_from_env(provider),
                note: Some(error.to_string()),
            }
        }
    };

    let response = transport
        .send(build_models_request(&config))
        .await
        .and_then(|response| extract_provider_models(provider, &response.body));

    match response {
        Ok(mut models) if !models.is_empty() => {
            // The configured default must always be selectable, even when the
            // provider's discovery response no longer ranks it into the
            // curated shortlist.
            if !models.iter().any(|model| model.id == config.model) {
                models.insert(0, LlmProviderModel { id: config.model.clone() });
            }
            LlmProviderCatalog {
                provider,
                enabled: true,
                default_model: Some(config.model),
                models,
                note: None,
            }
        }
        Ok(_) | Err(_) => LlmProviderCatalog {
            provider,
            enabled: true,
            default_model: Some(config.model),
            models: fallback_models_from_env(provider),
            note: Some("Using environment-configured model options because provider model discovery was unavailable.".to_string()),
        },
    }
}

fn fallback_models_from_env(provider: LlmProviderKind) -> Vec<LlmProviderModel> {
    let (list_var, default_var) = match provider {
        LlmProviderKind::OpenAi => ("TRUCO_OPENAI_MODELS", "TRUCO_OPENAI_MODEL"),
        LlmProviderKind::Anthropic => ("TRUCO_ANTHROPIC_MODELS", "TRUCO_ANTHROPIC_MODEL"),
        LlmProviderKind::OpenRouter => ("TRUCO_OPENROUTER_MODELS", "TRUCO_OPENROUTER_MODEL"),
    };

    let mut models = std::env::var(list_var)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|entry| !entry.is_empty())
                .map(|id| LlmProviderModel { id: id.to_string() })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if let Ok(default_model) = std::env::var(default_var) {
        if !models.iter().any(|model| model.id == default_model) {
            models.insert(0, LlmProviderModel { id: default_model });
        }
    }

    models
}

fn build_provider_request(config: &ProviderConfig, request: &LlmBotRequest) -> HttpRequest {
    let user_content = serde_json::to_string_pretty(&json!({
        "instruction": request.user_prompt,
        "turn": request.turn,
    }))
    .expect("llm bot request should serialize");

    match config.kind {
        // OpenAI and OpenRouter share the chat-completions wire format.
        LlmProviderKind::OpenAi | LlmProviderKind::OpenRouter => {
            let mut headers = vec![
                (
                    "authorization".to_string(),
                    format!("Bearer {}", config.api_key),
                ),
                ("content-type".to_string(), "application/json".to_string()),
            ];
            // OpenRouter uses these for app attribution / leaderboard ranking;
            // both are optional.
            if config.kind == LlmProviderKind::OpenRouter {
                if let Ok(referer) = std::env::var("TRUCO_OPENROUTER_REFERER") {
                    headers.push(("HTTP-Referer".to_string(), referer));
                }
                headers.push((
                    "X-Title".to_string(),
                    std::env::var("TRUCO_OPENROUTER_TITLE")
                        .unwrap_or_else(|_| OPENROUTER_DEFAULT_TITLE.to_string()),
                ));
            }
            HttpRequest {
                method: HttpMethod::Post,
                url: config.chat_url.clone(),
                headers,
                body: Some(json!({
                    "model": config.model,
                    "temperature": 0.9,
                    "response_format": { "type": "json_object" },
                    "messages": [
                        { "role": "system", "content": request.system_prompt },
                        { "role": "user", "content": user_content }
                    ]
                })),
            }
        }
        LlmProviderKind::Anthropic => HttpRequest {
            method: HttpMethod::Post,
            url: config.chat_url.clone(),
            headers: vec![
                ("x-api-key".to_string(), config.api_key.clone()),
                ("anthropic-version".to_string(), "2023-06-01".to_string()),
                ("content-type".to_string(), "application/json".to_string()),
            ],
            body: Some(json!({
                "model": config.model,
                "max_tokens": 800,
                "system": request.system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": user_content,
                    }
                ]
            })),
        },
    }
}

fn build_models_request(config: &ProviderConfig) -> HttpRequest {
    match config.kind {
        LlmProviderKind::OpenAi | LlmProviderKind::OpenRouter => HttpRequest {
            method: HttpMethod::Get,
            url: config.models_url.clone(),
            headers: vec![(
                "authorization".to_string(),
                format!("Bearer {}", config.api_key),
            )],
            body: None,
        },
        LlmProviderKind::Anthropic => HttpRequest {
            method: HttpMethod::Get,
            url: config.models_url.clone(),
            headers: vec![
                ("x-api-key".to_string(), config.api_key.clone()),
                ("anthropic-version".to_string(), "2023-06-01".to_string()),
            ],
            body: None,
        },
    }
}

fn extract_provider_content(kind: LlmProviderKind, raw: &str) -> Result<String, LlmProviderError> {
    let value: Value = serde_json::from_str(raw)
        .map_err(|error| LlmProviderError::InvalidResponse(error.to_string()))?;

    match kind {
        LlmProviderKind::OpenAi | LlmProviderKind::OpenRouter => value["choices"][0]["message"]
            ["content"]
            .as_str()
            .map(ToString::to_string)
            .ok_or_else(|| {
                LlmProviderError::InvalidResponse(
                    "OpenAI-compatible response did not include choices[0].message.content"
                        .to_string(),
                )
            }),
        LlmProviderKind::Anthropic => value["content"]
            .as_array()
            .and_then(|chunks| {
                chunks.iter().find_map(|chunk| {
                    (chunk["type"].as_str() == Some("text"))
                        .then(|| chunk["text"].as_str().map(ToString::to_string))
                        .flatten()
                })
            })
            .ok_or_else(|| {
                LlmProviderError::InvalidResponse(
                    "Anthropic response did not include a text content block".to_string(),
                )
            }),
    }
}

fn extract_provider_models(
    kind: LlmProviderKind,
    raw: &str,
) -> Result<Vec<LlmProviderModel>, LlmProviderError> {
    let value: Value = serde_json::from_str(raw)
        .map_err(|error| LlmProviderError::InvalidResponse(error.to_string()))?;
    let models = value["data"]
        .as_array()
        .or_else(|| value["models"].as_array())
        .ok_or_else(|| {
            LlmProviderError::InvalidResponse(
                "Provider model response did not include a model list".to_string(),
            )
        })?;

    let ids = models.iter().filter_map(|entry| entry["id"].as_str());

    if kind == LlmProviderKind::OpenRouter {
        // The OpenRouter models endpoint is queried with `sort=top-weekly`, so
        // the response is already ordered by recent popularity. Dedup keeping
        // the first occurrence, then curate in two passes: guarantee each
        // major-lab family up to OPENROUTER_FAMILY_LIMIT slots first, backfill
        // the rest by popularity. Selected entries are emitted in the original
        // popularity order regardless of which pass picked them.
        let mut seen = std::collections::HashSet::new();
        let ordered = ids
            .filter(|id| seen.insert(id.to_string()))
            .collect::<Vec<_>>();

        let mut selected = vec![false; ordered.len()];
        let mut total = 0usize;
        for prefix in OPENROUTER_FAMILY_PREFIXES {
            let mut taken = 0usize;
            for (index, id) in ordered.iter().enumerate() {
                if taken == OPENROUTER_FAMILY_LIMIT || total == OPENROUTER_MODEL_LIMIT {
                    break;
                }
                if !selected[index] && id.starts_with(prefix) {
                    selected[index] = true;
                    taken += 1;
                    total += 1;
                }
            }
        }
        for slot in selected.iter_mut() {
            if total == OPENROUTER_MODEL_LIMIT {
                break;
            }
            if !*slot {
                *slot = true;
                total += 1;
            }
        }

        let parsed = ordered
            .into_iter()
            .enumerate()
            .filter(|(index, _)| selected[*index])
            .map(|(_, id)| LlmProviderModel { id: id.to_string() })
            .collect::<Vec<_>>();
        return Ok(parsed);
    }

    let mut parsed = ids
        .map(|id| LlmProviderModel { id: id.to_string() })
        .collect::<Vec<_>>();
    parsed.sort_by(|left, right| left.id.cmp(&right.id));
    parsed.dedup_by(|left, right| left.id == right.id);
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::{
        build_provider_request, extract_provider_content, extract_provider_models, BoxFuture,
        HttpMethod, HttpRequest, HttpResponse, LlmProviderBot, LlmProviderCatalog,
        LlmProviderError, LlmProviderKind, LlmTransport, ProviderConfig,
    };
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use truco_engine::{Card, Hands, Match, Rank, Score, Suit, Turnup};

    #[derive(Clone)]
    struct FakeTransport {
        sent_requests: Arc<Mutex<Vec<HttpRequest>>>,
        response: HttpResponse,
    }

    impl LlmTransport for FakeTransport {
        fn send(
            &self,
            request: HttpRequest,
        ) -> BoxFuture<'_, Result<HttpResponse, LlmProviderError>> {
            let sent_requests = self.sent_requests.clone();
            let response = self.response.clone();
            Box::pin(async move {
                sent_requests
                    .lock()
                    .expect("requests mutex should lock")
                    .push(request);
                Ok(response)
            })
        }
    }

    fn sample_match() -> Match {
        let mut game = Match::new(1, Score { zero: 0, one: 0 }).expect("match should initialize");
        game.start_hand(
            Turnup {
                rank: Rank::Ace,
                suit: Suit::Spades,
            },
            Hands {
                zero: smallvec::smallvec![
                    Card {
                        id: "p0c0".into(),
                        rank: Rank::Seven,
                        suit: Suit::Diamonds,
                    },
                    Card {
                        id: "p0c1".into(),
                        rank: Rank::Six,
                        suit: Suit::Clubs,
                    },
                    Card {
                        id: "p0c2".into(),
                        rank: Rank::Four,
                        suit: Suit::Hearts,
                    },
                ],
                one: smallvec::smallvec![
                    Card {
                        id: "p1c0".into(),
                        rank: Rank::Three,
                        suit: Suit::Clubs,
                    },
                    Card {
                        id: "p1c1".into(),
                        rank: Rank::Five,
                        suit: Suit::Spades,
                    },
                    Card {
                        id: "p1c2".into(),
                        rank: Rank::Four,
                        suit: Suit::Diamonds,
                    },
                ],
            },
        )
        .expect("hand should start");
        game
    }

    #[test]
    fn openai_request_uses_json_chat_shape() {
        let request = build_provider_request(
            &ProviderConfig {
                kind: LlmProviderKind::OpenAi,
                model: "test-model".to_string(),
                api_key: "key".to_string(),
                chat_url: "https://example.test/chat".to_string(),
                models_url: "https://example.test/models".to_string(),
            },
            &truco_bot_core::build_llm_request(
                &truco_bot_core::turn_for_match(&sample_match(), 0).expect("turn should build"),
            ),
        );

        assert!(matches!(request.method, HttpMethod::Post));
        assert_eq!(request.url, "https://example.test/chat");
        assert_eq!(
            request.body.expect("body should exist")["model"],
            json!("test-model")
        );
    }

    #[test]
    fn anthropic_content_extraction_reads_text_blocks() {
        let content = extract_provider_content(
            LlmProviderKind::Anthropic,
            r#"{
                "content": [
                    { "type": "text", "text": "{\"reasoning\":\"brief\",\"choices\":[{\"action\":{\"type\":\"fold\"},\"weight\":1.0}]}" }
                ]
            }"#,
        )
        .expect("content should extract");

        assert!(content.contains("\"choices\""));
    }

    #[tokio::test]
    async fn provider_llm_bot_chooses_a_legal_action() {
        let sent_requests = Arc::new(Mutex::new(Vec::new()));
        let transport = FakeTransport {
            sent_requests: sent_requests.clone(),
            response: HttpResponse {
                body: r#"{
                    "choices": [
                        {
                            "message": {
                                "content": "{\"reasoning\":\"brief\",\"choices\":[{\"action\":{\"type\":\"play_face_up\",\"card_id\":\"p0c2\"},\"weight\":1.0}]}"
                            }
                        }
                    ]
                }"#
                .to_string(),
            },
        };
        let mut bot = LlmProviderBot::from_config(
            transport,
            ProviderConfig {
                kind: LlmProviderKind::OpenAi,
                model: "test-model".to_string(),
                api_key: "key".to_string(),
                chat_url: "https://example.test/chat".to_string(),
                models_url: "https://example.test/models".to_string(),
            },
            Some(9),
        )
        .expect("bot should build");

        let game = sample_match();
        let action = bot.choose_action(&game, 0).await.expect("bot should act");
        assert_eq!(
            action,
            truco_engine::Action::PlayFaceUp {
                card_id: "p0c2".into(),
            }
        );
        assert_eq!(
            sent_requests
                .lock()
                .expect("requests mutex should lock")
                .len(),
            1
        );
        assert!(bot.last_decision().is_some());
    }

    #[test]
    fn provider_catalog_type_serializes_stably() {
        let catalog = LlmProviderCatalog {
            provider: LlmProviderKind::OpenAi,
            enabled: true,
            default_model: Some("gpt-test".to_string()),
            models: vec![super::LlmProviderModel {
                id: "gpt-test".into(),
            }],
            note: None,
        };

        let value = serde_json::to_value(catalog).expect("catalog should serialize");
        assert_eq!(value["provider"], json!("openai"));
    }

    #[test]
    fn openrouter_kind_serializes_as_openrouter() {
        let value =
            serde_json::to_value(LlmProviderKind::OpenRouter).expect("kind should serialize");
        assert_eq!(value, json!("openrouter"));
    }

    #[test]
    fn openrouter_request_uses_bearer_auth_and_attribution() {
        let request = build_provider_request(
            &ProviderConfig {
                kind: LlmProviderKind::OpenRouter,
                model: "x-ai/grok-2".to_string(),
                api_key: "or-key".to_string(),
                chat_url: "https://openrouter.test/chat".to_string(),
                models_url: "https://openrouter.test/models".to_string(),
            },
            &truco_bot_core::build_llm_request(
                &truco_bot_core::turn_for_match(&sample_match(), 0).expect("turn should build"),
            ),
        );

        assert!(matches!(request.method, HttpMethod::Post));
        assert_eq!(request.url, "https://openrouter.test/chat");
        assert_eq!(
            request.body.as_ref().expect("body should exist")["model"],
            json!("x-ai/grok-2")
        );
        // OpenAI-style Bearer auth and a JSON response format.
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| name == "authorization" && value == "Bearer or-key"));
        assert_eq!(
            request.body.expect("body should exist")["response_format"]["type"],
            json!("json_object")
        );
        // App attribution header is always present for OpenRouter.
        assert!(request.headers.iter().any(|(name, _)| name == "X-Title"));
    }

    #[test]
    fn openrouter_models_preserve_popularity_order_and_cap() {
        // Deliberately out of alphabetical order, with a duplicate, to prove the
        // API's popularity order survives instead of being re-sorted.
        let entries = (0..30)
            .map(|index| json!({ "id": format!("zzz/model-{index:02}") }))
            .collect::<Vec<_>>();
        let mut data = vec![
            json!({ "id": "x-ai/grok-2" }),
            json!({ "id": "anthropic/claude" }),
            json!({ "id": "x-ai/grok-2" }), // duplicate of the first
        ];
        data.extend(entries);
        let raw = serde_json::to_string(&json!({ "data": data })).unwrap();

        let models = extract_provider_models(LlmProviderKind::OpenRouter, &raw)
            .expect("models should parse");

        assert_eq!(models.len(), super::OPENROUTER_MODEL_LIMIT);
        assert_eq!(models[0].id, "x-ai/grok-2");
        assert_eq!(models[1].id, "anthropic/claude");
        // the duplicate is dropped, so the third slot is the first zzz entry
        assert_eq!(models[2].id, "zzz/model-00");
    }

    #[test]
    fn openrouter_models_guarantee_family_coverage() {
        // 30 popular indie models push every major-lab model past the cap;
        // the family quota must still pull the labs into the shortlist.
        let mut data = (0..30)
            .map(|index| json!({ "id": format!("indie/model-{index:02}") }))
            .collect::<Vec<_>>();
        for id in [
            "openai/gpt-test",
            "anthropic/claude-test",
            "google/gemini-test",
            "deepseek/deepseek-test",
            "z-ai/glm-test",
        ] {
            data.push(json!({ "id": id }));
        }
        let raw = serde_json::to_string(&json!({ "data": data })).unwrap();

        let models = extract_provider_models(LlmProviderKind::OpenRouter, &raw)
            .expect("models should parse");
        let ids = models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>();

        assert_eq!(ids.len(), super::OPENROUTER_MODEL_LIMIT);
        // Every seeded family survives despite ranking below the cap.
        for id in [
            "openai/gpt-test",
            "anthropic/claude-test",
            "google/gemini-test",
            "deepseek/deepseek-test",
            "z-ai/glm-test",
        ] {
            assert!(ids.contains(&id), "missing family model {id}");
        }
        // Backfill keeps overall popularity order: the most popular indie
        // models fill the remaining 20 slots, in order, from the top.
        assert_eq!(ids[0], "indie/model-00");
        assert_eq!(ids[19], "indie/model-19");
        assert!(!ids.contains(&"indie/model-20"));
    }

    #[test]
    fn openai_models_stay_alphabetical() {
        let raw = serde_json::to_string(&json!({
            "data": [ { "id": "gpt-b" }, { "id": "gpt-a" }, { "id": "gpt-a" } ]
        }))
        .unwrap();
        let models =
            extract_provider_models(LlmProviderKind::OpenAi, &raw).expect("models should parse");
        assert_eq!(
            models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            vec!["gpt-a", "gpt-b"]
        );
    }

    #[test]
    fn byok_override_key_is_used_without_env() {
        // A supplied model + key resolves with no environment configuration.
        let config = ProviderConfig::from_env_with_key(
            LlmProviderKind::OpenRouter,
            Some("x-ai/grok-2".to_string()),
            Some("byok-secret".to_string()),
        )
        .expect("byok config should resolve");
        assert_eq!(config.api_key, "byok-secret");
        assert_eq!(config.model, "x-ai/grok-2");
        assert_eq!(config.chat_url, super::OPENROUTER_CHAT_COMPLETIONS_URL);
    }

    #[test]
    fn blank_byok_key_does_not_satisfy_config() {
        // A whitespace-only override must not be accepted as a real key; with no
        // env key set this surfaces as MissingConfig rather than a blank Bearer.
        std::env::remove_var("OPENROUTER_API_KEY");
        let result = ProviderConfig::from_env_with_key(
            LlmProviderKind::OpenRouter,
            Some("x-ai/grok-2".to_string()),
            Some("   ".to_string()),
        );
        assert!(matches!(result, Err(LlmProviderError::MissingConfig(_))));
    }
}
