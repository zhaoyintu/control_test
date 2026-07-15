#ifndef ADRC_CONTROLLER_H
#define ADRC_CONTROLLER_H

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * ADRC 控制器 (2nd-order ESO for 1st-order integrating plant) — 纯 C
 * ================================================================ */

typedef struct {
    double b0;       /**< 高频增益 (≈K)         */
    double wc;       /**< 控制器带宽 [rad/s]     */
    double wo;       /**< 观测器带宽 [rad/s]     */
    double wr;       /**< 参考轨迹带宽 [rad/s]   */
    int    nd;       /**< 死区延迟步数           */
    int    nd_max;   /**< 缓冲区最大长度         */

    /* 内部状态 */
    double z1;
    double z2;
    double *ubuf;    /**< 控制量延迟缓冲区       */
    int    ubuf_len;
    double rv1;
    double rv2;
} AdrcController;

/**
 * 初始化 ADRC 控制器
 * @param nd 死区延迟步数 (>=1)
 */
void adrc_init(AdrcController *adrc, double b0, double wc, double wo, double wr, int nd);

/**
 * 销毁 ADRC 控制器 (释放内部缓冲区)
 */
void adrc_free(AdrcController *adrc);

/**
 * ADRC 控制步进函数 (符合 ControllerStepFn 签名)
 */
double adrc_step(void *state, double y_meas, double sv_raw, double dt, int first);

#ifdef __cplusplus
}
#endif

#endif /* ADRC_CONTROLLER_H */