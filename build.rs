use std::env;
use std::path::PathBuf;

fn main() {
    build_myrustllm_cpu();
    build_myrustllm_cuda();
}

fn build_myrustllm_cpu() {
    println!("cargo:rerun-if-changed=myrustllm-cpu/CMakeLists.txt");
    println!("cargo:rerun-if-changed=myrustllm-cpu/src");
    println!("cargo:rerun-if-changed=myrustllm-cpu/include");

    let dst = cmake::build("myrustllm-cpu");
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=myrustllm-cpu");

    let bindings = bindgen::Builder::default()
        .header("myrustllm-cpu/include/myrustllm-cpu.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("cpu_tensor_.*")
        .allowlist_type("CPUTensor")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("myrustllm-cpu-bindings.rs"))
        .expect("Couldn't write bindings!");
}


fn build_myrustllm_cuda() {
    println!("cargo:rerun-if-changed=myrustllm-cuda/CMakeLists.txt");
    println!("cargo:rerun-if-changed=myrustllm-cuda/src");
    println!("cargo:rerun-if-changed=myrustllm-cuda/include");

    let dst = cmake::build("myrustllm-cuda");
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=myrustllm-cuda");

    let bindings = bindgen::Builder::default()
        .header("myrustllm-cuda/include/myrustllm-cuda.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .clang_arg(
            env::var("CUDA_HOME")
                .map(|s| format!("-I{}/include", s))
                .expect("No CUDA detected! Did you set $CUDA_HOME?"),
        )
        .allowlist_function("cuda_tensor_.*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("myrustllm-cuda-bindings.rs"))
        .expect("Couldn't write bindings!");
}
