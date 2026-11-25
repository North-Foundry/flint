#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <jpeglib.h>
#include <jerror.h>

typedef int (*write_output_fn)(void *, const uint8_t *, size_t);

struct flint_jpeg {
    struct jpeg_compress_struct compressor;
    struct jpeg_error_mgr error;
    struct jpeg_destination_mgr destination;
    write_output_fn write_output;
    jmp_buf jump;
    char message[JMSG_LENGTH_MAX];
    JOCTET buffer[8192];
};

static void error_exit(j_common_ptr compressor) {
    struct flint_jpeg *session = (struct flint_jpeg *)compressor;
    (*compressor->err->format_message)(compressor, session->message);
    longjmp(session->jump, 1);
}

static void reset_destination(j_compress_ptr compressor) {
    struct flint_jpeg *session = (struct flint_jpeg *)compressor;
    session->destination.next_output_byte = session->buffer;
    session->destination.free_in_buffer = sizeof(session->buffer);
}

static void write_destination(struct flint_jpeg *session, size_t length) {
    if (length != 0 &&
        !session->write_output(session->compressor.client_data, session->buffer, length)) {
        /* The Rust callback has returned, so this jump crosses only C frames. */
        ERREXIT(&session->compressor, JERR_FILE_WRITE);
    }
}

static boolean empty_destination(j_compress_ptr compressor) {
    struct flint_jpeg *session = (struct flint_jpeg *)compressor;
    write_destination(session, sizeof(session->buffer));
    reset_destination(compressor);
    return TRUE;
}

static void finish_destination(j_compress_ptr compressor) {
    struct flint_jpeg *session = (struct flint_jpeg *)compressor;
    write_destination(session, sizeof(session->buffer) - session->destination.free_in_buffer);
}

struct flint_jpeg *flint_jpeg_new(write_output_fn write_output) {
    struct flint_jpeg *session = calloc(1, sizeof(*session));
    if (session != NULL) {
        session->write_output = write_output;
        session->compressor.err = jpeg_std_error(&session->error);
        session->error.error_exit = error_exit;
        session->destination.init_destination = reset_destination;
        session->destination.empty_output_buffer = empty_destination;
        session->destination.term_destination = finish_destination;
    }
    return session;
}

void flint_jpeg_destroy(struct flint_jpeg *session) {
    jpeg_destroy_compress(&session->compressor);
    free(session);
}

const char *flint_jpeg_error(const struct flint_jpeg *session) {
    return session->message;
}

uint32_t flint_jpeg_max_dimension(void) {
    return JPEG_MAX_DIMENSION;
}

int flint_jpeg_start(struct flint_jpeg *session, void *output,
                     uint32_t width, uint32_t height, int components, int quality) {
    j_compress_ptr compressor = &session->compressor;
    if (setjmp(session->jump)) {
        compressor->client_data = NULL;
        return 0;
    }
    jpeg_create_compress(compressor);
    compressor->dest = &session->destination;
    compressor->client_data = output;
    compressor->image_width = width;
    compressor->image_height = height;
    compressor->input_components = components;
    compressor->in_color_space = components == 1 ? JCS_GRAYSCALE : JCS_RGB;

    /* Single-scan Huffman coding avoids full-image coefficient storage. */
    jpeg_c_set_int_param(compressor, JINT_COMPRESS_PROFILE, JCP_FASTEST);
    jpeg_set_defaults(compressor);
    compressor->optimize_coding = FALSE;
    if (components == 3) {
        compressor->comp_info[0].h_samp_factor = 2;
        compressor->comp_info[0].v_samp_factor = 2;
        compressor->comp_info[1].h_samp_factor = 1;
        compressor->comp_info[1].v_samp_factor = 1;
        compressor->comp_info[2].h_samp_factor = 1;
        compressor->comp_info[2].v_samp_factor = 1;
    }
    jpeg_set_quality(compressor, quality, TRUE);
    jpeg_start_compress(compressor, TRUE);
    compressor->client_data = NULL;
    return 1;
}

int flint_jpeg_write_row(struct flint_jpeg *session, void *output, const uint8_t *row) {
    j_compress_ptr compressor = &session->compressor;
    if (setjmp(session->jump)) {
        compressor->client_data = NULL;
        return 0;
    }
    compressor->client_data = output;
    /* The libjpeg API takes a mutable pointer but only reads the scanline. */
    JSAMPROW scanline = (JSAMPROW)row;
    if (jpeg_write_scanlines(compressor, &scanline, 1) != 1) {
        ERREXIT(compressor, JERR_CANT_SUSPEND);
    }
    compressor->client_data = NULL;
    return 1;
}

int flint_jpeg_finish(struct flint_jpeg *session, void *output) {
    j_compress_ptr compressor = &session->compressor;
    if (setjmp(session->jump)) {
        compressor->client_data = NULL;
        return 0;
    }
    compressor->client_data = output;
    jpeg_finish_compress(compressor);
    compressor->client_data = NULL;
    return 1;
}
