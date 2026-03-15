import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";

interface UserSettings {
  llm_provider: string;
  api_key: string;
  profession: string;
  coding_style: string;
  default_tone: string;
}

function App() {
  const [settings, setSettings] = useState<UserSettings>({
    llm_provider: "gemini",
    api_key: "",
    profession: "",
    coding_style: "",
    default_tone: "",
  });
  
  const [statusMsg, setStatusMsg] = useState("");

  useEffect(() => {
    // Load initial settings from SQLite backend
    invoke<UserSettings>("get_settings")
      .then((data) => setSettings(data))
      .catch((err) => console.error("Failed to load settings:", err));
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    setSettings({
      ...settings,
      [e.target.name]: e.target.value,
    });
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await invoke("save_settings", { payload: settings });
      setStatusMsg("Settings saved successfully.");
      setTimeout(() => setStatusMsg(""), 3000);
    } catch (error) {
      setStatusMsg("Error saving settings.");
      console.error(error);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <div className="logo">&gt;_</div>
        <div className="title">prompt enhancer</div>
      </div>

      <form onSubmit={handleSave}>
        <div className="form-group">
          <label>ai provider</label>
          <select name="llm_provider" value={settings.llm_provider} onChange={handleChange}>
            <option value="gemini">Google Gemini</option>
            <option value="groq">Groq</option>
          </select>
        </div>

        <div className="form-group">
          <label>api key</label>
          <input
            type="password"
            name="api_key"
            value={settings.api_key}
            onChange={handleChange}
            placeholder="sk-..."
          />
        </div>

        <div className="form-group">
          <label>profession / context</label>
          <input
            type="text"
            name="profession"
            value={settings.profession}
            onChange={handleChange}
            placeholder="e.g. Python Backend Developer"
          />
        </div>

        <div className="form-group">
          <label>coding style rules</label>
          <textarea
            name="coding_style"
            value={settings.coding_style}
            onChange={handleChange}
            placeholder="e.g. Use type hints. Write clean, concise code."
          />
        </div>

        <div className="form-group">
          <label>default tone</label>
          <input
            type="text"
            name="default_tone"
            value={settings.default_tone}
            onChange={handleChange}
            placeholder="e.g. Professional and direct."
          />
        </div>

        <div className="button-container">
          <button type="submit" className="primary">
            save settings
          </button>
        </div>
      </form>
      
      <div className="status-message">{statusMsg}</div>
    </div>
  );
}

export default App;
