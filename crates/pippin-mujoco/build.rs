//! Locate MuJoCo, generate bindings, and link against its shared library.
//!
//! Search order: $MUJOCO_DIR (containing include/mujoco/mujoco.h and the
//! library), then the `mujoco` Python package of $PYTHON, $VIRTUAL_ENV, or
//! python3 on PATH.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn from_python(python: &str) -> Option<PathBuf> {
    let out = Command::new(python)
        .args(["-c", "import mujoco, os; print(os.path.dirname(mujoco.__file__))"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(PathBuf::from(String::from_utf8(out.stdout).ok()?.trim()))
}

fn find_root() -> PathBuf {
    if let Ok(dir) = env::var("MUJOCO_DIR") {
        return PathBuf::from(dir);
    }
    let mut pythons = vec![];
    if let Ok(p) = env::var("PYTHON") {
        pythons.push(p);
    }
    if let Ok(v) = env::var("VIRTUAL_ENV") {
        pythons.push(format!("{v}/bin/python"));
    }
    // workspace-local venv
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    pythons.push(manifest.join("../../.venv/bin/python").to_string_lossy().into_owned());
    pythons.push("python3".into());
    for p in &pythons {
        if let Some(dir) = from_python(p) {
            return dir;
        }
    }
    panic!("MuJoCo not found: set MUJOCO_DIR or `pip install mujoco` (tried {pythons:?})");
}

fn find_lib(root: &Path) -> PathBuf {
    for dir in [root.to_path_buf(), root.join("lib")] {
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.starts_with("libmujoco") && (name.ends_with(".dylib") || name.contains(".so")) {
                    return e.path();
                }
            }
        }
    }
    panic!("libmujoco not found under {}", root.display());
}

/// Copy the library into OUT_DIR as `libmujoco.dylib` with an absolute
/// install name, so every binary linking this crate (tests, examples, the
/// Python extension) finds it without rpath setup. Re-signed ad hoc because
/// editing the install name invalidates the signature on Apple Silicon.
fn stage_library(lib: &Path) -> PathBuf {
    let dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("lib");
    std::fs::create_dir_all(&dir).unwrap();
    let ext = if cfg!(target_os = "macos") { "dylib" } else { "so" };
    let dst = dir.join(format!("libmujoco.{ext}"));
    std::fs::copy(lib, &dst).unwrap();
    if cfg!(target_os = "macos") {
        let run = |cmd: &str, args: &[&str]| {
            let ok = Command::new(cmd).args(args).status().map(|s| s.success()).unwrap_or(false);
            assert!(ok, "{cmd} {args:?} failed");
        };
        let d = dst.to_string_lossy().into_owned();
        run("install_name_tool", &["-id", &d, &d]);
        run("codesign", &["-f", "-s", "-", &d]);
    }
    dir
}

fn main() {
    println!("cargo:rerun-if-env-changed=MUJOCO_DIR");
    println!("cargo:rerun-if-changed=build.rs");
    let root = find_root();
    let include = root.join("include");
    let lib = find_lib(&root);
    let dir = stage_library(&lib);
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=dylib=mujoco");
    println!("cargo:libdir={}", dir.display());

    let bindings = bindgen::Builder::default()
        .header(include.join("mujoco/mujoco.h").to_string_lossy())
        .clang_arg(format!("-I{}", include.display()))
        .allowlist_function("mj_.*|mju_.*|mjs_.*")
        .allowlist_type("mj.*")
        .allowlist_var("mj.*")
        .layout_tests(false)
        .derive_default(false)
        .generate()
        .expect("bindgen failed on mujoco.h");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out.join("mujoco.rs")).unwrap();
}
