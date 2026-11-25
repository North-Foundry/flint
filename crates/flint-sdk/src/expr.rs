use crate::{Error, ImageInfo, PixelFormat, Result};
use std::{
    iter::Sum,
    ops::{Add, Div, Mul, Sub},
    sync::Arc,
};

/// Largest accepted horizontal or vertical offset in a stencil, in pixels.
pub const MAX_STENCIL_OFFSET: i32 = 4096;
const MAX_PROGRAM_LENGTH: usize = 65_536;
const MAX_STAGES: usize = 128;

/// A postfix instruction. A sample without a channel uses the active channel.
#[derive(Clone, Copy, Debug)]
pub enum Instruction {
    Constant(f32),
    Sample {
        dx: i32,
        dy: i32,
        channel: Option<usize>,
    },
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

/// A scalar symbolic expression. Its operations run on normalized f32 samples.
#[derive(Clone, Debug)]
pub struct Expr(Vec<Instruction>);

impl Expr {
    fn sample(dx: i32, dy: i32, channel: Option<usize>) -> Self {
        Self(vec![Instruction::Sample { dx, dy, channel }])
    }

    fn binary(mut self, mut rhs: Self, operator: Instruction) -> Self {
        if self.0.is_empty() || rhs.0.is_empty() || self.0.len() + rhs.0.len() >= MAX_PROGRAM_LENGTH
        {
            return Self(Vec::new());
        }
        self.0.append(&mut rhs.0);
        self.0.push(operator);
        self
    }

    /// Selects the smaller operand, preferring a number over NaN and returning positive zero for zero.
    pub fn min(self, rhs: impl Into<Self>) -> Self {
        self.binary(rhs.into(), Instruction::Min)
    }

    /// Selects the larger operand, preferring a number over NaN and returning positive zero for zero.
    pub fn max(self, rhs: impl Into<Self>) -> Self {
        self.binary(rhs.into(), Instruction::Max)
    }

    /// Evaluates max(self, low) followed by min(result, high).
    pub fn clamp(self, low: impl Into<Self>, high: impl Into<Self>) -> Self {
        self.max(low).min(high)
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.0
    }
}

impl From<f32> for Expr {
    fn from(value: f32) -> Self {
        // Quiet NaNs make number-preferring min and max agree across CPU backends.
        let value = if value.is_nan() { f32::NAN } else { value };
        Self(vec![Instruction::Constant(value)])
    }
}

macro_rules! arithmetic {
    ($trait:ident, $method:ident, $opcode:ident) => {
        impl<T: Into<Expr>> $trait<T> for Expr {
            type Output = Expr;
            fn $method(self, rhs: T) -> Expr {
                self.binary(rhs.into(), Instruction::$opcode)
            }
        }

        impl $trait<Expr> for f32 {
            type Output = Expr;
            fn $method(self, rhs: Expr) -> Expr {
                Expr::from(self).$method(rhs)
            }
        }
    };
}

arithmetic!(Add, add, Add);
arithmetic!(Sub, sub, Sub);
arithmetic!(Mul, mul, Mul);
arithmetic!(Div, div, Div);

impl Sum for Expr {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(Add::add).unwrap_or_else(|| 0.0.into())
    }
}

/// Symbolic RGB channel access. Gray input broadcasts its sample to all three fields.
pub struct PixelSamples {
    pub r: Expr,
    pub g: Expr,
    pub b: Expr,
}

/// The output of a pixelwise operation.
pub struct Pixel {
    format: PixelFormat,
    channels: Vec<Expr>,
}

impl Pixel {
    pub fn gray(value: impl Into<Expr>) -> Self {
        Self {
            format: PixelFormat::Gray8,
            channels: vec![value.into()],
        }
    }

    pub fn rgb(r: impl Into<Expr>, g: impl Into<Expr>, b: impl Into<Expr>) -> Self {
        Self {
            format: PixelFormat::Rgb8,
            channels: vec![r.into(), g.into(), b.into()],
        }
    }
}

/// Symbolic relative access to samples of the active input channel.
pub struct Neighborhood {
    _private: (),
}

impl Neighborhood {
    pub fn at(&self, dx: i32, dy: i32) -> Expr {
        Expr::sample(dx, dy, None)
    }
}

pub trait Operation: Send + Sync {
    fn apply(&self, image: ImageExpr) -> Result<ImageExpr>;
}

#[derive(Clone, Copy, Debug)]
pub enum Primitive {
    MapColor,
    MapPixel,
    Stencil,
}

/// Read-only graph node. The current runtime supports one input per operation.
pub enum Node {
    Source,
    Stage(Stage),
}

pub struct Stage {
    input: ImageExpr,
    primitive: Primitive,
    programs: Vec<Expr>,
}

impl Stage {
    pub fn input(&self) -> &ImageExpr {
        &self.input
    }
    pub fn primitive(&self) -> Primitive {
        self.primitive
    }
    pub fn programs(&self) -> &[Expr] {
        &self.programs
    }
}

struct ImageNode {
    node: Node,
    info: ImageInfo,
    depth: usize,
}

/// An immutable image graph. Clones share the graph and carry no decoded pixels.
#[derive(Clone)]
pub struct ImageExpr(Arc<ImageNode>);

impl ImageExpr {
    pub fn source(info: ImageInfo) -> Self {
        Self(Arc::new(ImageNode {
            node: Node::Source,
            info,
            depth: 0,
        }))
    }

    pub fn info(&self) -> &ImageInfo {
        &self.0.info
    }

    pub fn node(&self) -> &Node {
        &self.0.node
    }

    /// Identity of the source node is preserved when an operation appends stages.
    pub fn shares_source(&self, other: &Self) -> bool {
        fn root(mut image: &ImageExpr) -> &ImageExpr {
            while let Node::Stage(stage) = image.node() {
                image = stage.input();
            }
            image
        }
        Arc::ptr_eq(&root(self).0, &root(other).0)
    }

    /// Build one expression that the runtime evaluates independently for every channel.
    pub fn map_color(self, build: impl FnOnce(Expr) -> Expr) -> Result<Self> {
        let format = self.info().format();
        self.stage(
            Primitive::MapColor,
            format,
            vec![build(Expr::sample(0, 0, None))],
        )
    }

    /// Build output channels from the input pixel's named channels.
    pub fn map_pixel(self, build: impl FnOnce(PixelSamples) -> Pixel) -> Result<Self> {
        let pixel = build(PixelSamples {
            r: Expr::sample(0, 0, Some(0)),
            g: Expr::sample(0, 0, Some(1)),
            b: Expr::sample(0, 0, Some(2)),
        });
        self.stage(Primitive::MapPixel, pixel.format, pixel.channels)
    }

    /// Build one expression using clamp-to-edge spatial samples in the active channel.
    pub fn stencil(self, build: impl FnOnce(Neighborhood) -> Expr) -> Result<Self> {
        let format = self.info().format();
        self.stage(
            Primitive::Stencil,
            format,
            vec![build(Neighborhood { _private: () })],
        )
    }

    fn stage(self, primitive: Primitive, format: PixelFormat, programs: Vec<Expr>) -> Result<Self> {
        let depth = self.0.depth + 1;
        if depth > MAX_STAGES {
            return Err(Error::Limit(format!(
                "at most {MAX_STAGES} processing stages are supported"
            )));
        }

        for program in &programs {
            if program.0.is_empty() {
                return Err(Error::Limit(format!(
                    "expression exceeds {MAX_PROGRAM_LENGTH} instructions"
                )));
            }
            for &instruction in program.instructions() {
                if let Instruction::Sample { dx, dy, channel } = instruction {
                    let allowed = match primitive {
                        Primitive::Stencil => {
                            channel.is_none()
                                && dx.unsigned_abs() <= MAX_STENCIL_OFFSET as u32
                                && dy.unsigned_abs() <= MAX_STENCIL_OFFSET as u32
                        }
                        Primitive::MapColor => channel.is_none() && dx == 0 && dy == 0,
                        Primitive::MapPixel => channel.is_some() && dx == 0 && dy == 0,
                    };
                    if !allowed {
                        return Err(Error::Invalid("sample reference is outside the primitive's allowed neighborhood or channel scope".into()));
                    }
                }
            }
        }

        let info = self.info().with_format(format)?;
        Ok(Self(Arc::new(ImageNode {
            node: Node::Stage(Stage {
                input: self,
                primitive,
                programs,
            }),
            info,
            depth,
        })))
    }
}
