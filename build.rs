use std::{env, fs, path::Path};
fn main() {
    println!("cargo:rerun-if-changed=webui");
    let out = env::var("OUT_DIR").unwrap();
    let mut code = String::from("pub static WEBUI_ASSETS: &[(&str, &str, &[u8])] = &[\n");
    fn mime_for(p: &str) -> &'static str {
        match p.rsplit('.').next().unwrap_or("") {
            "html" => "text/html; charset=utf-8",
            "js" => "text/javascript; charset=utf-8",
            "css" => "text/css; charset=utf-8",
            "svg" => "image/svg+xml",
            "woff2" => "font/woff2",
            "woff" => "font/woff",
            "json" => "application/json",
            "png" => "image/png",
            "ico" => "image/x-icon",
            "map" => "application/json",
            _ => "application/octet-stream",
        }
    }
    fn walk(dir: &Path, base: &Path, code: &mut String) {
        for e in fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, base, code);
                continue;
            }
            let rel = p
                .strip_prefix(base)
                .unwrap()
                .to_str()
                .unwrap()
                .replace('\\', "/");
            code.push_str(&format!(
                "  ({:?}, {:?}, include_bytes!({:?})),\n",
                rel,
                mime_for(&rel),
                p.to_str().unwrap()
            ));
        }
    }
    // Resolve webui/ to an absolute path so include_bytes!() (which resolves
    // relative to $OUT_DIR/webui_assets.rs) finds the files at the crate root.
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
    let webui = Path::new(&manifest).join("webui");
    if webui.exists() {
        walk(&webui, &webui, &mut code);
    }
    code.push_str("];\n");
    fs::write(Path::new(&out).join("webui_assets.rs"), code).unwrap();
}
