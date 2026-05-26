use starkbot_api::engine::StarkbotEngine;
use starkbot_api::{Backend, BackendConfig, FrontendCommand};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    dotenvy::dotenv().ok();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let no_auto_approve = args.iter().any(|a| a == "--no-auto-approve");
    let persona_slug = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .map(|s| s.as_str())
        .unwrap_or("starkbot");

    let config = BackendConfig {
        persona_slug: persona_slug.to_string(),
        api_key: String::new(),
        model_name: String::new(),
        auto_approve: !no_auto_approve,
    };

    let mut engine = StarkbotEngine::new(config)?;
    let handle = engine.start().await?;

    log::info!(
        "starkbot-daemon started (persona={}, auto_approve={})",
        persona_slug,
        !no_auto_approve
    );

    let mut events = handle.events;
    tokio::spawn(async move {
        while events.recv().await.is_some() {}
    });

    tokio::signal::ctrl_c().await?;
    log::info!("starkbot-daemon shutting down");

    let _ = handle.commands.send(FrontendCommand::Shutdown);
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    Ok(())
}
