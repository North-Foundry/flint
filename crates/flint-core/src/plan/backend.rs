/*!
 * Backend selection and host SIMD detection.
 */

/**
 * Scalar reference execution, automatic CPU selection, or SIMD with scalar fallback.
 */
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Backend {
    /**
     * Select SIMD when the host supports it.
     */
    #[default]
    Auto,
    /**
     * Use the scalar reference evaluator.
     */
    Scalar,
    /**
     * Request SIMD, falling back to scalar when it is unavailable.
     */
    Simd,
}

impl Backend {
    /**
     * Report whether this resolved backend will evaluate any four-pixel groups.
     */
    pub(crate) fn uses_vectors(self, width: u32) -> bool {
        self == Self::Simd && width >= 4
    }

    /**
     * Resolve automatic selection and fall back to scalar when SIMD is unavailable.
     */
    pub fn resolved(self) -> Self {
        if self != Self::Scalar && simd_available() {
            Self::Simd
        } else {
            Self::Scalar
        }
    }
}

/**
 * Report whether the compiled target and the running host provide the SIMD
 * baseline used by `wide`.
 */
fn simd_available() -> bool {
    // wide uses explicit SIMD when these baseline features are compiled in.
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ))]
    {
        std::arch::is_x86_feature_detected!("sse2")
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        ),
        all(target_arch = "aarch64", target_feature = "neon"),
    )))]
    {
        false
    }
}
