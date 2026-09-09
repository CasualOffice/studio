//! Emits vault-format test vectors as JSON so the Python engine
//! implementation can be checked against this one byte for byte.
//!
//! Run: cargo run --example vault_vectors

use melp_model_studio_lib::vault_test_support as v;

fn main() {
    let file_key: [u8; 32] = std::array::from_fn(|i| (i * 7 + 3) as u8);
    let file_id: [u8; 16] = std::array::from_fn(|i| (i * 11 + 5) as u8);

    // Sizes that straddle the chunk boundary, where an off-by-one in the
    // final-chunk flag or the counter would show up.
    let sizes = [0usize, 1, 100, 262_143, 262_144, 262_145, 700_000];

    let mut out = Vec::new();
    for len in sizes {
        let pt: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        let sealed = v::seal_with_file_key(&file_key, &file_id, &pt);
        out.push(serde_json::json!({
            "len": len,
            "sealed_hex": hex(&sealed),
            "sealed_len": sealed.len(),
        }));
    }

    println!(
        "{}",
        serde_json::json!({
            "file_key_hex": hex(&file_key),
            "file_id_hex": hex(&file_id),
            "vectors": out,
        })
    );
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
