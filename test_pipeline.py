"""Quick end-to-end test for the unified KeyDNA pipeline."""
import sys, numpy as np
sys.path.insert(0, '.')
from core.features import FeatureExtractor
from core.capture import SimulatedCapture
from models.auth_model import UnifiedAuthModel

cap = SimulatedCapture()
ext = FeatureExtractor()
model = UnifiedAuthModel()

# ── Enrollment ──
print("=== ENROLLMENT ===")
for i in range(10):
    events = cap.simulate_typing(
        avg_speed_ms=100, variance_ms=15, dwell_ms=80,
        n_keys=10, error_rate=0.02, seed=42 + i)
    features = ext.extract(events)
    ok = model.enroll(features)
    print(f"  Sample {i+1}: dim={len(features)}, enrolled={ok}")

print(f"Model ready: {model.is_ready}, samples: {model.sample_count}")
assert model.is_ready, "Model should be ready after 10 samples"
assert model.sample_count == 10

# ── Genuine user tests (should ACCEPT) ──
print("\n=== GENUINE USER TEST ===")
print(f"  Threshold Accept: {model._threshold_accept:.4f}")
print(f"  Threshold Retry:  {model._threshold_retry:.4f}")
genuine_accepts = 0
for i in range(5):
    events = cap.simulate_typing(
        avg_speed_ms=100, variance_ms=15, dwell_ms=80,
        n_keys=10, error_rate=0.02, seed=200 + i)
    features = ext.extract(events)
    model.reset_attack_tracking()
    
    # Manually compute score to see what it is
    score = float(model._model.decision_function(features.reshape(1, -1))[0])
    
    result = model.authenticate(features)
    decision = result["decision"]
    print(f"  Test {i+1}: {decision} (Score: {score:.4f})")
    if decision == "ACCEPT":
        genuine_accepts += 1

print(f"  Genuine acceptance rate: {genuine_accepts}/5")

# ── Impostor tests (different speed/rhythm - should REJECT) ──
print("\n=== IMPOSTOR TEST ===")
impostor_rejects = 0
for i in range(5):
    events = cap.simulate_typing(
        avg_speed_ms=60, variance_ms=30, dwell_ms=50,
        n_keys=10, error_rate=0.08, seed=300 + i)
    features = ext.extract(events)
    model.reset_attack_tracking()
    
    score = float(model._model.decision_function(features.reshape(1, -1))[0])
    
    result = model.authenticate(features)
    decision = result["decision"]
    print(f"  Test {i+1}: {decision} (Score: {score:.4f})")
    if decision != "ACCEPT":
        impostor_rejects += 1

print(f"  Impostor rejection rate: {impostor_rejects}/5")

# ── Replay attack test (consistency > 0.97) ──
print("\n=== REPLAY ATTACK TEST ===")
events = cap.simulate_typing(
    avg_speed_ms=100, variance_ms=15, dwell_ms=80,
    n_keys=10, error_rate=0.02, seed=42)
features = ext.extract(events)
features[11] = 0.985  # Fake replay consistency
model.reset_attack_tracking()
result = model.authenticate(features)
print(f"  Decision: {result['decision']}")
print(f"  Replay detected: {result['replay_detected']}")
assert result["replay_detected"], "Replay attack should be detected"
assert result["decision"] == "REJECT", "Replay should be REJECTED"

# ── Feature vector size verification ──
print("\n=== FEATURE VECTOR VERIFICATION ===")
events = cap.simulate_typing(
    avg_speed_ms=100, variance_ms=15, dwell_ms=80,
    n_keys=10, error_rate=0.02, seed=999)
features = ext.extract(events)
print(f"  Feature vector size: {len(features)}")
print(f"  Expected: {ext.feature_dim}")
assert len(features) == 27, f"Expected 27 features, got {len(features)}"
print(f"  Global features [0:17]: {features[:17].shape}")
print(f"  Sequence features [17:27]: {features[17:27].shape}")

# ── Information hiding check ──
print("\n=== INFORMATION HIDING CHECK ===")
events = cap.simulate_typing(
    avg_speed_ms=60, variance_ms=30, dwell_ms=50,
    n_keys=10, error_rate=0.08, seed=400)
features = ext.extract(events)
model.reset_attack_tracking()
result = model.authenticate(features)
assert "confidence" not in result, "Result must NOT contain confidence"
assert result["reason"] == "Authentication failed.", \
    f"Rejection reason must be generic, got: {result['reason']}"
print(f"  Result keys: {list(result.keys())}")
print(f"  Reason: {result['reason']}")
print(f"  No confidence exposed: PASS")

# ── Serialization test ──
print("\n=== SERIALIZATION TEST ===")
data = model.get_samples_serializable()
print(f"  Serialized {len(data)} samples")
model2 = UnifiedAuthModel()
model2.load_samples(data)
assert model2.sample_count == 10
print(f"  Deserialized: {model2.sample_count} samples, ready={model2.is_ready}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED OK")
print("=" * 50)
