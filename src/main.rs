use arboard::Clipboard;
use enigo::{Enigo, Key, KeyboardControllable};
use global_hotkey::{
    hotkey::{Code, HotKey, Modifiers},
    GlobalHotKeyEvent, GlobalHotKeyManager,
};
use reqwest::Client;
use rusqlite::{Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use async_trait::async_trait;

#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn enhance_prompt(&self, api_key: &str, user_context: &str, raw_text: &str) -> Option<String>;
}

// ---------------------------------------------------------
// Gemini Provider Implementation
// ---------------------------------------------------------
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    system_instruction: Content,
    contents: Vec<Content>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Part {
    text: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct GeminiResponse {
    candidates: Option<Vec<Candidate>>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Candidate {
    content: Content,
}

pub struct GeminiProvider;

#[async_trait]
impl LlmProvider for GeminiProvider {
    async fn enhance_prompt(&self, api_key: &str, user_context: &str, raw_text: &str) -> Option<String> {
        let client = Client::new();
        let system_prompt = format!(
            "You are an expert AI prompt engineer. The user will provide a rough draft of a prompt they want to send to another AI. Your job is to rewrite and greatly improve their prompt to make it clear, detailed, and highly effective. Do not answer their prompt or write the code they are asking for. Only return the final, improved prompt text. Do not include introductory or concluding text like 'Here is the improved prompt'.\n\nTake the following user profile context into consideration when improving the prompt:\n\n{}",
            user_context
        );

        let request_body = GeminiRequest {
            system_instruction: Content {
                parts: vec![Part { text: system_prompt }],
            },
            contents: vec![
                Content {
                    parts: vec![Part { text: raw_text.to_string() }],
                }
            ],
        };

        println!("[Gemini] Sending request to Gemini API...");
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={}",
            api_key
        );

        let res = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await;

        match res {
            Ok(response) => {
                if response.status().is_success() {
                    if let Ok(parsed) = response.json::<GeminiResponse>().await {
                        if let Some(candidates) = parsed.candidates {
                            if let Some(first) = candidates.first() {
                                if let Some(part) = first.content.parts.first() {
                                    return Some(part.text.clone());
                                }
                            }
                        }
                    }
                } else {
                    println!("[Gemini] API Error: {:?}", response.text().await);
                }
            }
            Err(e) => println!("[Gemini] Request failed: {:?}", e),
        }
        None
    }
}

// ---------------------------------------------------------
// Groq Provider Implementation (OpenAI Format)
// ---------------------------------------------------------
#[derive(Serialize, Deserialize, Debug)]
struct GroqMessage {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct GroqRequest {
    model: String,
    messages: Vec<GroqMessage>,
}

#[derive(Serialize, Deserialize, Debug)]
struct GroqChoice {
    message: GroqMessage,
}

#[derive(Serialize, Deserialize, Debug)]
struct GroqResponse {
    choices: Vec<GroqChoice>,
}

pub struct GroqProvider;

#[async_trait]
impl LlmProvider for GroqProvider {
    async fn enhance_prompt(&self, api_key: &str, user_context: &str, raw_text: &str) -> Option<String> {
        let client = Client::new();
        let system_prompt = format!(
            "You are an expert AI prompt engineer. The user will provide a rough draft of a prompt they want to send to another AI. Your job is to rewrite and greatly improve their prompt to make it clear, detailed, and highly effective. Do not answer their prompt or write the code they are asking for. Only return the final, improved prompt text. Do not include introductory or concluding text like 'Here is the improved prompt'.\n\nTake the following user profile context into consideration when improving the prompt:\n\n{}",
            user_context
        );

        let request_body = GroqRequest {
            model: "llama-3.1-8b-instant".to_string(), // Groq free tier
            messages: vec![
                GroqMessage { role: "system".to_string(), content: system_prompt },
                GroqMessage { role: "user".to_string(), content: raw_text.to_string() }
            ],
        };

        println!("[Groq] Sending request to Groq API...");

        let res = client
            .post("https://api.groq.com/openai/v1/chat/completions")
            .bearer_auth(api_key)
            .json(&request_body)
            .send()
            .await;

        match res {
            Ok(response) => {
                if response.status().is_success() {
                    if let Ok(parsed) = response.json::<GroqResponse>().await {
                        if let Some(choice) = parsed.choices.first() {
                            return Some(choice.message.content.clone());
                        }
                    }
                } else {
                    println!("[Groq] API Error: {:?}", response.text().await);
                }
            }
            Err(e) => println!("[Groq] Request failed: {:?}", e),
        }
        None
    }
}

/// Gets the path to the local SQLite database file.
fn get_db_path() -> PathBuf {
    let mut path = env::var_os("APPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let mut p = PathBuf::from(env::var("HOME").unwrap_or_else(|_| ".".to_string()));
            p.push(".config");
            p
        });
    path.push("prompt-enhancer");
    if !path.exists() {
        fs::create_dir_all(&path).expect("Failed to create app data directory");
    }
    path.push("user_data.db");
    path
}

fn init_db() -> SqlResult<Connection> {
    let db_path = get_db_path();
    println!("Initializing database at: {:?}", db_path);
    let conn = Connection::open(db_path)?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY,
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL
        )",
        [],
    )?;
    Ok(conn)
}

fn seed_dummy_profile(conn: &Connection) -> SqlResult<()> {
    println!("Seeding dummy user profile data...");
    let dummy_data = [
        ("profession", "Python Developer"),
        ("coding_style", "Use type hints. Write clean, concise code. Avoid unnecessary comments."),
        ("default_tone", "Professional and direct."),
    ];
    for (key, value) in dummy_data.iter() {
        conn.execute(
            "INSERT INTO user_profile (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [key, value],
        )?;
    }
    Ok(())
}

fn get_user_context(conn: &Connection) -> String {
    let mut stmt = conn.prepare("SELECT key, value FROM user_profile").unwrap();
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }).unwrap();

    let mut context = String::from("User Profile Details:\n");
    for row in rows {
        if let Ok((k, v)) = row {
            context.push_str(&format!("- {}: {}\n", k, v));
        }
    }
    context
}

use std::sync::mpsc;
use winit::event_loop::{ControlFlow, EventLoopBuilder};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::window::WindowId;

fn capture_selected_text() -> Option<String> {
    let mut enigo = Enigo::new();
    
    // Give the user a moment to organically release the physical hotkeys (Ctrl+Shift+Space)
    // before we simulate keystrokes. Otherwise, 'C' becomes 'Ctrl+Shift+C' (Inspect Element).
    thread::sleep(Duration::from_millis(400));
    
    // Explicitly release modifier keys logically, to prevent bleeding
    enigo.key_up(Key::Shift);
    enigo.key_up(Key::Control);
    enigo.key_up(Key::Alt);
    
    // Simulate Ctrl+C
    enigo.key_down(Key::Control);
    enigo.key_click(Key::Raw(0x43)); // 'c'
    enigo.key_up(Key::Control);
    
    // Wait briefly to allow the OS to update the clipboard
    thread::sleep(Duration::from_millis(150));
    
    if let Ok(mut clipboard) = Clipboard::new() {
        if let Ok(text) = clipboard.get_text() {
            return Some(text);
        }
    }
    None
}

fn paste_replaced_text(text: &str) {
    if let Ok(mut clipboard) = Clipboard::new() {
        let _ = clipboard.set_text(text.to_string());
        
        thread::sleep(Duration::from_millis(100)); // allow clipboard update
        
        let mut enigo = Enigo::new();
        // Clear modifiers
        enigo.key_up(Key::Shift);
        enigo.key_up(Key::Control);
        enigo.key_up(Key::Alt);
        
        enigo.key_down(Key::Control);
        enigo.key_click(Key::Raw(0x56)); // 'v'
        enigo.key_up(Key::Control);
    }
}

// enhance_prompt logic is now inside LlmProvider impls

// ---------------------------------------------------------
// Winit Application Handler
// ---------------------------------------------------------
struct App {
    conn: Connection,
    api_key: String,
    hotkey_id: u32,
    provider: std::sync::Arc<dyn LlmProvider>,
    receiver: Option<mpsc::Receiver<Option<String>>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {}

    fn window_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: WindowId,
        _event: WindowEvent,
    ) {}

    fn suspended(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {}
    
    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // Check if we have a pending AI response
        if let Some(rx) = &self.receiver {
            match rx.try_recv() {
                Ok(Some(enhanced_text)) => {
                    println!("Enhanced Text Received, Replacing...");
                    paste_replaced_text(&enhanced_text);
                    self.receiver = None;
                }
                Ok(None) => {
                    println!("Failed to get enhancement from API.");
                    self.receiver = None;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    // Still waiting... do nothing
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.receiver = None;
                }
            }
        }

        // Only poll for new hotkey events if we aren't currently waiting for an AI response
        if self.receiver.is_none() {
            if let Ok(event) = GlobalHotKeyEvent::receiver().try_recv() {
                if event.id == self.hotkey_id {
                    println!("Hotkey triggered!");
                    
                    if let Some(selected_text) = capture_selected_text() {
                        println!("Captured Text: {}", selected_text);
                        
                        let context = get_user_context(&self.conn);
                        let api_key = self.api_key.clone();
                        let provider = std::sync::Arc::clone(&self.provider);
                        
                        // Spawn logic in a background thread 
                        let (tx, rx) = mpsc::channel();
                        self.receiver = Some(rx);
                        
                        std::thread::spawn(move || {
                            let rt = tokio::runtime::Runtime::new().unwrap();
                            let enhanced = rt.block_on(provider.enhance_prompt(&api_key, &context, &selected_text));
                            let _ = tx.send(enhanced);
                        });
                    } else {
                        println!("Failed to capture text.");
                    }
                }
            }
        }
    }
}

fn main() -> SqlResult<()> {
    println!("Starting Prompt Enhancer Backend...");
    let conn = init_db()?;
    seed_dummy_profile(&conn)?;
    
    // Determine provider from config
    let provider_name = env::var("LLM_PROVIDER").unwrap_or_else(|_| "gemini".to_string());
    
    let provider: std::sync::Arc<dyn LlmProvider> = match provider_name.to_lowercase().as_str() {
        "groq" => std::sync::Arc::new(GroqProvider),
        _ => std::sync::Arc::new(GeminiProvider), // default to gemini
    };
    
    println!("Using LLM Provider: {}", provider_name.to_lowercase());
    
    // Ensure the correct API key is present based on the selected provider
    let api_key_var = if provider_name.to_lowercase() == "groq" { "GROQ_API_KEY" } else { "GEMINI_API_KEY" };
    let api_key = env::var(api_key_var).unwrap_or_else(|_| {
        println!("WARNING: {} environment variable not set. API calls will fail.", api_key_var);
        String::new()
    });

    let manager = GlobalHotKeyManager::new().expect("Failed to create hotkey manager");
    let hotkey1 = HotKey::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Space);
    manager.register(hotkey1).expect("Failed to register hotkey");
    
    println!("Listening for global hotkey: Ctrl + Shift + Space...");
    
    // Initialize standard Winit Event Loop for Windows GUI messages
    let event_loop = EventLoopBuilder::new().build().unwrap();
    event_loop.set_control_flow(ControlFlow::WaitUntil(
        std::time::Instant::now() + Duration::from_millis(50),
    ));
    
    let mut app = App {
        conn,
        api_key,
        hotkey_id: hotkey1.id,
        provider,
        receiver: None,
    };
    
    let _ = event_loop.run_app(&mut app);
    
    Ok(())
}
