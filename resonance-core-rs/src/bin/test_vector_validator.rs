//! Test vector validator binary

use resonance_core::ResonanceEncryption;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::    println!("\n\nValidation Results:");
    println!("Algorithm: {} {}", wrapper.metadata.algorithm, wrapper.metadata.test_type);
    println!("Total vectors: {}", results.total_vectors);
    println!("Passed: {}", results.passed);
    println!("Failed: {}", results.failed.len());Buf;

#[derive(Debug, Deserialize)]
struct TestVectorWrapper {
    metadata: TestVectorMetadata,
    vectors: Vec<TestVector>,
}

#[derive(Debug, Deserialize)]
struct TestVectorMetadata {
    algorithm: String,
    #[serde(rename = "type")]
    test_type: String,
    timestamp: String,
    count: usize,
    format_version: String,
}

#[derive(Debug, Deserialize)]
struct TestVector {
    id: usize,
    #[serde(rename = "type")]
    vector_type: String,
    key_hex: String,
    plaintext_hex: String,
    ciphertext_hex: String,
    plaintext_size: usize,
    key_size: usize,
}

// Add field aliases for compatibility
impl TestVector {
    fn key(&self) -> &str {
        &self.key_hex
    }
    
    fn plaintext(&self) -> &str {
        &self.plaintext_hex
    }
    
    fn ciphertext(&self) -> &str {
        &self.ciphertext_hex
    }
}

#[derive(Debug, Serialize)]
struct ValidationResult {
    total_vectors: usize,
    passed: usize,
    failed: Vec<TestFailure>,
}

#[derive(Debug, Serialize)]
struct TestFailure {
    vector_index: usize,
    expected: String,
    actual: String,
    description: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path_to_test_vectors.json>", args[0]);
        std::process::exit(1);
    }

    let vector_path = PathBuf::from(&args[1]);
    println!("Loading test vectors from: {}", vector_path.display());
    
    let file = File::open(&vector_path)?;
    let reader = BufReader::new(file);
    
    // Parse the wrapper structure
    let wrapper: TestVectorWrapper = serde_json::from_reader(reader)?;
    
    println!("Loaded test vector file: {} {} ({})", 
             wrapper.metadata.algorithm, 
             wrapper.metadata.test_type,
             wrapper.metadata.timestamp);
    println!("Contains {} vectors", wrapper.vectors.len());
    
    let mut results = ValidationResult {
        total_vectors: wrapper.vectors.len(),
        passed: 0,
        failed: Vec::new(),
    };
    
    for (i, vector) in wrapper.vectors.iter().enumerate() {
        print!("\rTesting vector {}/{}", i + 1, wrapper.vectors.len());
        
        // Create a debug print of the test vector
        println!("\nVector #{} (id: {}) details:", i, vector.id);
        println!("  Type: {}", vector.vector_type);
        println!("  Key (hex): {}", vector.key());
        println!("  Plaintext (hex): {}", vector.plaintext());
        println!("  Expected ciphertext (hex): {}", vector.ciphertext());
        
    // Convert hex key to bytes then use it
    let key_bytes = match hex::decode(&vector.key) {
        Ok(bytes) => bytes,
        Err(e) => {
            results.failed.push(TestFailure {
                vector_index: i,
                expected: vector.key.clone(),
                actual: format!("Error decoding key: {}", e),
                description: "Invalid key hex format".into(),
            });
            continue;
        }
    };
        
    // Use the key bytes 
    let enc = ResonanceEncryption::from_raw_key(&key_bytes);
        
    // Convert hex to bytes
    let plaintext = hex::decode(&vector.plaintext)?;
    let expected_ciphertext = hex::decode(&vector.ciphertext)?;
        
        // Test encryption
        let encrypted = match enc.encrypt(&plaintext) {
            Ok(e) => e,
            Err(e) => {
                results.failed.push(TestFailure {
                    vector_index: i,
                    expected: vector.ciphertext.clone(),
                    actual: format!("Error: {}", e),
                    description: "Encryption failed".into(),
                });
                continue;
            }
        };
        
        if encrypted != expected_ciphertext {
            results.failed.push(TestFailure {
                vector_index: i,
                expected: vector.ciphertext.clone(),
                actual: hex::encode(&encrypted),
                description: "Ciphertext mismatch".into(),
            });
            continue;
        }
        
        // Test decryption
        let decrypted = match enc.decrypt(&expected_ciphertext) {
            Ok(d) => d,
            Err(e) => {
                results.failed.push(TestFailure {
                    vector_index: i,
                    expected: vector.plaintext.clone(),
                    actual: format!("Error: {}", e),
                    description: "Decryption failed".into(),
                });
                continue;
            }
        };
        
        if decrypted != plaintext {
            results.failed.push(TestFailure {
                vector_index: i,
                expected: vector.plaintext.clone(),
                actual: hex::encode(&decrypted),
                description: "Plaintext mismatch".into(),
            });
            continue;
        }
        
        results.passed += 1;
    }
    
    println!("\n\nValidation Results:");
    println!("Total vectors: {}", results.total_vectors);
    println!("Passed: {}", results.passed);
    println!("Failed: {}", results.failed.len());
    
    if !results.failed.is_empty() {
        println!("\nFailure Details:");
        for failure in &results.failed {
            println!("\nVector #{}", failure.vector_index);
            println!("Description: {}", failure.description);
            println!("Expected: {}", failure.expected);
            println!("Actual:   {}", failure.actual);
        }
    }
    
    // Save detailed results
    let results_path = vector_path.with_extension("results.json");
    serde_json::to_writer_pretty(
        File::create(&results_path)?,
        &results
    )?;
    println!("\nDetailed results saved to: {}", results_path.display());
    
    if !results.failed.is_empty() {
        std::process::exit(1);
    }
    
    Ok(())
}
