fn main() {
    let include =
        std::env::var_os("DEP_JPEG_INCLUDE").expect("mozjpeg-sys must supply its headers");
    println!("cargo:rerun-if-changed=src/bridge.c");
    println!("cargo:rerun-if-env-changed=DEP_JPEG_INCLUDE");
    cc::Build::new()
        .includes(std::env::split_paths(&include))
        .file("src/bridge.c")
        .compile("flint_jpeg_bridge");
}
