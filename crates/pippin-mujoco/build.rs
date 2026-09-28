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

/// Directory to put on the rpath so the library's install name resolves.
/// The pip wheel ships `libmujoco.X.dylib` flat while its install name is
/// `@rpath/mujoco.framework/Versions/A/libmujoco.X.dylib`; in that case build a
/// matching symlink tree in OUT_DIR.
fn rpath_for(lib: &Path, libdir: &Path) -> PathBuf {
    let install_name = Command::new("otool")
        .args(["-D", &lib.to_string_lossy()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().nth(1).map(|l| l.trim().to_string()));
    let Some(rel) = install_name.as_deref().and_then(|n| n.strip_prefix("@rpath/")) else {
        return libdir.to_path_buf();
    };
    if libdir.join(rel).exists() {
        return libdir.to_path_buf();
    }
    let root = PathBuf::from(env::var("OUT_DIR").unwrap()).join("rpath");
    let link = root.join(rel);
    std::fs::create_dir_all(link.parent().unwrap()).unwrap();
    let _ = std::fs::remove_file(&link);
    std::os::unix::fs::symlink(lib, &link).unwrap();
    root
}

fn main() {
    println!("cargo:rerun-if-env-changed=MUJOCO_DIR");
    println!("cargo:rerun-if-changed=build.rs");
    let root = find_root();
    let include = root.join("include");
    let lib = find_lib(&root);
    let libdir = lib.parent().unwrap();
    println!("cargo:rustc-link-arg={}", lib.display());
    let rpath = rpath_for(&lib, libdir);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath.display());
    println!("cargo:libdir={}", rpath.display());

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
