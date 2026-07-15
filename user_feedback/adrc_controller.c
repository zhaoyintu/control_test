#include "adrc_controller.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void adrc_init(AdrcController *a, double b0, double wc, double wo, double wr, int nd) {
    a->b0 = b0; a->wc = wc; a->wo = wo; a->wr = wr;
    a->nd = nd; a->nd_max = nd + 1;
    a->z1 = 0.0; a->z2 = 0.0;
    a->rv1 = 0.0; a->rv2 = 0.0;
    a->ubuf_len = a->nd_max;
    a->ubuf = (double *)calloc((size_t)a->ubuf_len, sizeof(double));
}

void adrc_free(AdrcController *a) {
    free(a->ubuf); a->ubuf = NULL;
    a->ubuf_len = 0;
}

double adrc_step(void *state, double y_meas, double sv_raw, double dt, int first) {
    AdrcController *a = (AdrcController *)state;

    if (first) {
        a->z1 = y_meas; a->z2 = 0.0;
        a->rv1 = sv_raw; a->rv2 = 0.0;
        for (int i = 0; i < a->ubuf_len; i++) a->ubuf[i] = 0.0;
    }

    /* 二阶临界阻尼 TD */
    double v1 = a->rv1, v2 = a->rv2;
    double v1n = v1 + dt * v2;
    double v2n = v2 + dt * (-2.0 * a->wr * v2 - a->wr * a->wr * (v1 - sv_raw));
    a->rv1 = v1n; a->rv2 = v2n;
    double sv = v1n;

    double z1 = a->z1, z2 = a->z2;
    double u_delayed = a->ubuf[a->nd];  /* ND 步前的控制量 */

    /* 2nd-order ESO */
    double err = y_meas - z1;
    double z1n = z1 + dt * (z2 + a->b0 * u_delayed + 2.0 * a->wo * err);
    double z2n = z2 + dt * (a->wo * a->wo * err);
    a->z1 = z1n; a->z2 = z2n;

    /* 控制律 (1st-order) */
    double u0 = a->wc * (sv - z1n);
    double u_cmd = (u0 - z2n) / a->b0;

    double us = u_cmd;
    if (us > 100.0) us = 100.0;
    if (us < 0.0)   us = 0.0;

    /* 更新延迟缓冲 (FIFO shift right) */
    for (int i = a->ubuf_len - 1; i > 0; i--) a->ubuf[i] = a->ubuf[i - 1];
    a->ubuf[0] = us;

    return us;
}