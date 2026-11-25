//! Contracts shared by Flint's runtime and independent extensions.

mod codec;
mod expr;

pub use codec::{
    Decoder, Destination, Encoder, ImageInformation, PixelFormat, Rows, RowsMut, Source,
};
pub use expr::{
    Expr, ImageExpression, MAX_STENCIL_OFFSET, Neighborhood, Operation, Pixel, PixelSamples,
};

/// Read-only execution contracts for runtimes consuming the SDK's graph.
pub mod graph {
    pub use crate::expr::{Instruction, Node, Primitive, Stage};
}

use std::{fmt, io};

/// Errors from codec I/O, graph construction, and execution.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Io(io::Error),
    Invalid(String),
    Unsupported(String),
    Codec(String),
    Limit(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(f),
            Self::Invalid(message) => write!(f, "invalid input: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported: {message}"),
            Self::Codec(message) => write!(f, "codec error: {message}"),
            Self::Limit(message) => write!(f, "resource limit: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
