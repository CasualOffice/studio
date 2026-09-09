// Keep the console window hidden in release builds on Windows; harmless on macOS.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    melp_model_studio_lib::run()
}
