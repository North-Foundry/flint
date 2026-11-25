use flint_sdk::{Decoder, Destination, Encoder, Error, ImageInfo, Result, Rows, RowsMut, Source};
use std::io::{Cursor, Read};

pub(crate) trait DecoderSession {
    fn info(&self) -> &ImageInfo;
    fn read_rows(&mut self, output: RowsMut<'_>) -> Result<usize>;
    fn finish(self: Box<Self>) -> Result<()>;
}

impl<T: Decoder> DecoderSession for T {
    fn info(&self) -> &ImageInfo {
        Decoder::info(self)
    }
    fn read_rows(&mut self, output: RowsMut<'_>) -> Result<usize> {
        Decoder::read_rows(self, output)
    }
    fn finish(self: Box<Self>) -> Result<()> {
        Decoder::finish(*self)
    }
}

pub(crate) trait EncoderSession {
    fn write_rows(&mut self, rows: Rows<'_>) -> Result<()>;
    fn finish(self: Box<Self>) -> Result<()>;
}

impl<T: Encoder> EncoderSession for T {
    fn write_rows(&mut self, rows: Rows<'_>) -> Result<()> {
        Encoder::write_rows(self, rows)
    }
    fn finish(self: Box<Self>) -> Result<()> {
        Encoder::finish(*self)
    }
}

fn open_decoder<T: Decoder>(source: Source) -> Result<Box<dyn DecoderSession>> {
    Ok(Box::new(T::open(source)?))
}

fn open_encoder<T: Encoder>(
    destination: Destination,
    info: &ImageInfo,
) -> Result<Box<dyn EncoderSession>> {
    Ok(Box::new(T::open(destination, info, T::Options::default())?))
}

#[derive(Clone, Copy)]
pub struct SelectedDecoder {
    name: &'static str,
    probe: fn(&[u8]) -> bool,
    open: OpenDecoder,
}

type OpenDecoder = fn(Source) -> Result<Box<dyn DecoderSession>>;

impl SelectedDecoder {
    pub fn name(self) -> &'static str {
        self.name
    }

    pub(crate) fn session(self, source: Source) -> Result<Box<dyn DecoderSession>> {
        (self.open)(source)
    }
}

type OpenEncoder = fn(Destination, &ImageInfo) -> Result<Box<dyn EncoderSession>>;

#[derive(Clone, Copy)]
pub struct SelectedEncoder {
    extensions: &'static [&'static str],
    open: OpenEncoder,
}

impl SelectedEncoder {
    pub(crate) fn session(
        self,
        destination: Destination,
        info: &ImageInfo,
    ) -> Result<Box<dyn EncoderSession>> {
        (self.open)(destination, info)
    }
}

#[derive(Default)]
pub struct Registry {
    decoders: Vec<SelectedDecoder>,
    encoders: Vec<SelectedEncoder>,
}

impl Registry {
    pub fn register_decoder<T: Decoder>(&mut self) {
        self.decoders.push(SelectedDecoder {
            name: T::NAME,
            probe: T::probe,
            open: open_decoder::<T>,
        });
    }

    pub fn register_encoder<T: Encoder>(&mut self) {
        self.encoders.push(SelectedEncoder {
            extensions: T::EXTENSIONS,
            open: open_encoder::<T>,
        });
    }

    pub fn inspect(&self, mut source: Source) -> Result<(SelectedDecoder, ImageInfo)> {
        let mut prefix = [0; 64];
        let mut len = 0;
        while len < prefix.len() {
            match source.read(&mut prefix[len..]) {
                Ok(0) => break,
                Ok(count) => len += count,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
        }
        let codec = self
            .decoders
            .iter()
            .rev()
            .find(|codec| (codec.probe)(&prefix[..len]))
            .copied()
            .ok_or_else(|| Error::Unsupported("unrecognized input format".into()))?;
        let session =
            codec.session(Box::new(Cursor::new(prefix).take(len as u64).chain(source)))?;
        Ok((codec, *session.info()))
    }

    pub fn encoder(&self, extension: &str) -> Result<SelectedEncoder> {
        self.encoders
            .iter()
            .rev()
            .find(|codec| {
                codec
                    .extensions
                    .iter()
                    .any(|candidate| extension.eq_ignore_ascii_case(candidate))
            })
            .copied()
            .ok_or_else(|| Error::Unsupported(format!("no encoder for extension {extension:?}")))
    }
}
