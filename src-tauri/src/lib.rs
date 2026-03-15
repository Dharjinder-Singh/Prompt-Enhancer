use arboard::Clipboard;
use enigo::{Enigo, Key, KeyboardControllable};
use reqwest::Client;
use rusqlite::{Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::Manager;
use async_trait::async_trait;
use tauri_plugin_global_shortcut::{Code, Modifiers, ShortcutState};

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

// ---------------------------------------------------------
// Database Setup
// ---------------------------------------------------------
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

// ---------------------------------------------------------
// Keystroke and Clipboard Management
// ---------------------------------------------------------
fn capture_selected_text() -> Option<String> {
    let mut enigo = Enigo::new();
    
    // Give the user a moment to organically release the physical hotkeys
    thread::sleep(Duration::from_millis(400));
    
    // Explicitly release modifier keys logically
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

// ---------------------------------------------------------
// Tauri State Management
// ---------------------------------------------------------
struct AppState {
    db: Mutex<Connection>,
    api_key: Mutex<String>,
    provider: Mutex<Arc<dyn LlmProvider>>,
    is_processing: Mutex<bool>,
}

#[derive(Serialize)]
struct UserSettings {
    llm_provider: String,
    api_key: String,
    profession: String,
    coding_style: String,
    default_tone: String,
}

#[derive(Deserialize)]
struct SaveSettingsPayload {
    llm_provider: String,
    api_key: String,
    profession: String,
    coding_style: String,
    default_tone: String,
}

#[tauri::command]
fn get_settings(state: tauri::State<AppState>) -> UserSettings {
    let db = state.db.lock().unwrap();
    let mut stmt = db.prepare("SELECT key, value FROM user_profile").unwrap();
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }).unwrap();

    let mut settings = UserSettings {
        llm_provider: "gemini".to_string(),
        api_key: "".to_string(),
        profession: "".to_string(),
        coding_style: "".to_string(),
        default_tone: "".to_string(),
    };

    for row in rows {
        if let Ok((k, v)) = row {
            match k.as_str() {
                "llm_provider" => settings.llm_provider = v,
                "api_key" => settings.api_key = v,
                "profession" => settings.profession = v,
                "coding_style" => settings.coding_style = v,
                "default_tone" => settings.default_tone = v,
                _ => {}
            }
        }
    }
    settings
}

#[tauri::command]
fn save_settings(state: tauri::State<AppState>, payload: SaveSettingsPayload) -> Result<(), String> {
    let db = state.db.lock().unwrap();
    
    let to_save = [
        ("llm_provider", payload.llm_provider.as_str()),
        ("api_key", payload.api_key.as_str()),
        ("profession", payload.profession.as_str()),
        ("coding_style", payload.coding_style.as_str()),
        ("default_tone", payload.default_tone.as_str()),
    ];
    
    for (key, value) in to_save.iter() {
        db.execute(
            "INSERT INTO user_profile (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [key, value],
        ).map_err(|e| e.to_string())?;
    }
    drop(db);
    
    // Hot-swap live engine State
    let new_provider: Arc<dyn LlmProvider> = match payload.llm_provider.to_lowercase().as_str() {
        "groq" => Arc::new(GroqProvider),
        _ => Arc::new(GeminiProvider),
    };
    
    let mut provider_lock = state.provider.lock().unwrap();
    *provider_lock = new_provider;
    
    let mut key_lock = state.api_key.lock().unwrap();
    *key_lock = payload.api_key.clone();
    
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let conn = init_db().expect("Failed to init db");
    
    // If table is empty, seed it
    {
        let mut stmt = conn.prepare("SELECT COUNT(*) FROM user_profile").unwrap();
        let count: i64 = stmt.query_row([], |row| row.get(0)).unwrap_or(0);
        if count == 0 {
            seed_dummy_profile(&conn).expect("Failed to seed db");
        }
    }

    // Load initial state from DB
    let mut llm_provider = "gemini".to_string();
    let mut api_key = String::new();
    
    {
        let mut stmt = conn.prepare("SELECT key, value FROM user_profile").unwrap();
        let rows: Vec<_> = stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))
            .unwrap()
            .collect();
            
        for row in rows {
            if let Ok((k, v)) = row {
                if k == "llm_provider" && !v.is_empty() { llm_provider = v.clone(); }
                if k == "api_key" { api_key = v.clone(); }
            }
        }
    }

    let provider: Arc<dyn LlmProvider> = match llm_provider.to_lowercase().as_str() {
        "groq" => Arc::new(GroqProvider),
        _ => Arc::new(GeminiProvider),
    };
    
    let app_state = AppState {
        db: Mutex::new(conn),
        api_key: Mutex::new(api_key),
        provider: Mutex::new(provider),
        is_processing: Mutex::new(false),
    };

    tauri::Builder::default()
        .manage(app_state)
        .plugin(tauri_plugin_opener::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        if shortcut.matches(Modifiers::CONTROL | Modifiers::SHIFT, Code::Space) {
                            let state = app.state::<AppState>();
                            
                            // Prevent overlapping triggers
                            let mut is_processing = state.is_processing.lock().unwrap();
                            if *is_processing {
                                return;
                            }
                            *is_processing = true;
                            
                            println!("Hotkey triggered!");
                            
                            if let Some(selected_text) = capture_selected_text() {
                                println!("Captured Text: {}", selected_text);
                                
                                let db = state.db.lock().unwrap();
                                let context = get_user_context(&db);
                                drop(db); // Release lock early
                                
                                let provider_lock = state.provider.lock().unwrap();
                                let provider = Arc::clone(&*provider_lock);
                                drop(provider_lock);
                                
                                let api_key_lock = state.api_key.lock().unwrap();
                                let api_key = api_key_lock.clone();
                                drop(api_key_lock);
                                
                                // Spawn background async task
                                let app_handle = app.clone();
                                tauri::async_runtime::spawn(async move {
                                    if let Some(enhanced_text) = provider.enhance_prompt(&api_key, &context, &selected_text).await {
                                        println!("Enhanced Text Received, Replacing...");
                                        paste_replaced_text(&enhanced_text);
                                    } else {
                                        println!("Failed to get enhancement from API.");
                                    }
                                    
                                    // Release processing lock
                                    let state = app_handle.state::<AppState>();
                                    let mut is_processing = state.is_processing.lock().unwrap();
                                    *is_processing = false;
                                });
                            } else {
                                println!("Failed to capture text.");
                                *is_processing = false;
                            }
                        }
                    }
                })
                .build()
        )
        .setup(|app| {
            // Register shortcut dynamically based on plugin
            use tauri_plugin_global_shortcut::{Shortcut, GlobalShortcutExt};
            app.global_shortcut().register(Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Space))?;
            
            // Build simple System Tray natively
            use tauri::menu::{Menu, MenuItem};
            use tauri::tray::TrayIconBuilder;
            
            let show_i = MenuItem::with_id(app, "show", "Open Settings", true, None::<&str>)?;
            let hide_i = MenuItem::with_id(app, "hide", "Hide Settings", true, None::<&str>)?;
            let quit_i = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show_i, &hide_i, &quit_i])?;

            let _tray = TrayIconBuilder::new()
                .menu(&menu)
                .icon(app.default_window_icon().unwrap().clone())
                .on_menu_event(|app_handle, event| {
                    match event.id.as_ref() {
                        "quit" => app_handle.exit(0),
                        "show" => {
                            if let Some(window) = app_handle.get_webview_window("main") {
                                let _ = window.show();
                                let _ = window.set_focus();
                            }
                        }
                        "hide" => {
                            if let Some(window) = app_handle.get_webview_window("main") {
                                let _ = window.hide();
                            }
                        }
                        _ => {}
                    }
                })
                .build(app)?;
                
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_settings, save_settings])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
