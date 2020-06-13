#include <algorithm>
#include <csignal>
#include <ctime>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "boost/iterator/counting_iterator.hpp"

#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
Dtype overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2) {
  float l1 = x1 - w1 / 2;
  float l2 = x2 - w2 / 2;
  float left = l1 > l2 ? l1 : l2;
  float r1 = x1 + w1 / 2;
  float r2 = x2 + w2 / 2;
  float right = r1 < r2 ? r1 : r2;
  return right - left;
}
template float overlap(float, float, float, float);
template double overlap(double, double, double, double);

template <typename Dtype>
Dtype box_intersection(const vector<Dtype> &a, const vector<Dtype> &b) {
  float w = overlap(a[0], a[2], b[0], b[2]);
  float h = overlap(a[1], a[3], b[1], b[3]);
  if (w < 0 || h < 0)
    return 0;
  float area = w * h;
  return area;
}
template float box_intersection(const vector<float> &a, const vector<float> &b);
template double box_intersection(const vector<double> &a,
                                 const vector<double> &b);

float box_intersection(const box &a, const box &b) {
  return box_intersection(a.to_vector<float>(), b.to_vector<float>());
}

template <typename Dtype>
Dtype box_union(const vector<Dtype> &a, const vector<Dtype> &b) {
  float i = box_intersection(a, b);
  float u = a[2] * a[3] + b[2] * b[3] - i;
  return u;
}
template float box_union(const vector<float> &a, const vector<float> &b);
template double box_union(const vector<double> &a, const vector<double> &b);
float box_union(const box &a, const box &b) {
  return box_union(a.to_vector<float>(), b.to_vector<float>());
}
template <typename Dtype>
Dtype box_iou(const vector<Dtype> &a, const vector<Dtype> &b) {
  return box_intersection(a, b) / box_union(a, b);
}
template float box_iou(const vector<float> &a, const vector<float> &b);
template double box_iou(const vector<double> &a, const vector<double> &b);
float box_iou(const box &a, const box &b) {
  return box_iou(a.to_vector<float>(), b.to_vector<float>());
}

template <typename Dtype>
Dtype box_iou(const vector<Dtype> &a, const vector<Dtype> &b, IOU_LOSS type) {
  Dtype iou;
  if (type == GIOU) {
    iou = box_giou(a, b);
  } else if (type == DIOU) {
    iou = box_diou(a, b);
  } else if (type == CIOU) {
    iou = box_ciou(a, b);
  } else {
    iou = box_iou(a, b);
  }
  return iou;
}
template float box_iou(const vector<float> &a, const vector<float> &b,
                       IOU_LOSS type);
template double box_iou(const vector<double> &a, const vector<double> &b,
                        IOU_LOSS type);

float box_iou(const box &a, const box &b, IOU_LOSS type) {
  return box_iou(a.to_vector<float>(), b.to_vector<float>(), type);
}
dxrep dx_box_iou(const box &pred, const box &truth, IOU_LOSS iou_loss) {
  boxabs pred_tblr = to_tblr(pred);
  float pred_t = fmin(pred_tblr.top, pred_tblr.bot);
  float pred_b = fmax(pred_tblr.top, pred_tblr.bot);
  float pred_l = fmin(pred_tblr.left, pred_tblr.right);
  float pred_r = fmax(pred_tblr.left, pred_tblr.right);

  boxabs truth_tblr = to_tblr(truth);
#ifdef DEBUG_PRINTS
  printf("\niou: %f, giou: %f\n", box_iou(pred, truth), box_giou(pred, truth));
  printf("pred: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)\n",
         pred.x, pred.y, pred.w, pred.h, pred_tblr.top, pred_tblr.bot,
         pred_tblr.left, pred_tblr.right);
  printf("truth: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)\n",
         truth.x, truth.y, truth.w, truth.h, truth_tblr.top, truth_tblr.bot,
         truth_tblr.left, truth_tblr.right);
#endif
  // printf("pred (t,b,l,r): (%f, %f, %f, %f)\n", pred_t, pred_b, pred_l,
  // pred_r); printf("trut (t,b,l,r): (%f, %f, %f, %f)\n", truth_tblr.top,
  // truth_tblr.bot, truth_tblr.left, truth_tblr.right);
  dxrep ddx = {0};
  float X = (pred_b - pred_t) * (pred_r - pred_l);
  float Xhat =
      (truth_tblr.bot - truth_tblr.top) * (truth_tblr.right - truth_tblr.left);
  float Ih = fmin(pred_b, truth_tblr.bot) - fmax(pred_t, truth_tblr.top);
  float Iw = fmin(pred_r, truth_tblr.right) - fmax(pred_l, truth_tblr.left);
  float I = Iw * Ih;
  float U = X + Xhat - I;
  float S = powf(pred.x - truth.x, 2) + powf(pred.y - truth.y, 2);
  float giou_Cw =
      fmax(pred_r, truth_tblr.right) - fmin(pred_l, truth_tblr.left);
  float giou_Ch = fmax(pred_b, truth_tblr.bot) - fmin(pred_t, truth_tblr.top);
  float giou_C = giou_Cw * giou_Ch;

  // float IoU = I / U;
  // Partial Derivatives, derivatives
  float dX_wrt_t = -1 * (pred_r - pred_l);
  float dX_wrt_b = pred_r - pred_l;
  float dX_wrt_l = -1 * (pred_b - pred_t);
  float dX_wrt_r = pred_b - pred_t;

  // gradient of I min/max in IoU calc (prediction)
  float dI_wrt_t = pred_t > truth_tblr.top ? (-1 * Iw) : 0;
  float dI_wrt_b = pred_b < truth_tblr.bot ? Iw : 0;
  float dI_wrt_l = pred_l > truth_tblr.left ? (-1 * Ih) : 0;
  float dI_wrt_r = pred_r < truth_tblr.right ? Ih : 0;
  // derivative of U with regard to x
  float dU_wrt_t = dX_wrt_t - dI_wrt_t;
  float dU_wrt_b = dX_wrt_b - dI_wrt_b;
  float dU_wrt_l = dX_wrt_l - dI_wrt_l;
  float dU_wrt_r = dX_wrt_r - dI_wrt_r;
  // gradient of C min/max in IoU calc (prediction)
  float dC_wrt_t = pred_t < truth_tblr.top ? (-1 * giou_Cw) : 0;
  float dC_wrt_b = pred_b > truth_tblr.bot ? giou_Cw : 0;
  float dC_wrt_l = pred_l < truth_tblr.left ? (-1 * giou_Ch) : 0;
  float dC_wrt_r = pred_r > truth_tblr.right ? giou_Ch : 0;

  // Final IOU loss (prediction) (negative of IOU gradient, we want the negative
  // loss)
  float p_dt = 0;
  float p_db = 0;
  float p_dl = 0;
  float p_dr = 0;
  if (U > 0) {
    p_dt = ((U * dI_wrt_t) - (I * dU_wrt_t)) / (U * U);
    p_db = ((U * dI_wrt_b) - (I * dU_wrt_b)) / (U * U);
    p_dl = ((U * dI_wrt_l) - (I * dU_wrt_l)) / (U * U);
    p_dr = ((U * dI_wrt_r) - (I * dU_wrt_r)) / (U * U);
  }

  // apply grad from prediction min/max for correct corner selection
  p_dt = pred_tblr.top < pred_tblr.bot ? p_dt : p_db;
  p_db = pred_tblr.top < pred_tblr.bot ? p_db : p_dt;
  p_dl = pred_tblr.left < pred_tblr.right ? p_dl : p_dr;
  p_dr = pred_tblr.left < pred_tblr.right ? p_dr : p_dl;

  if (iou_loss == GIOU) {
    if (giou_C > 0) {
      // apply "C" term from gIOU
      p_dt += ((giou_C * dU_wrt_t) - (U * dC_wrt_t)) / (giou_C * giou_C);
      p_db += ((giou_C * dU_wrt_b) - (U * dC_wrt_b)) / (giou_C * giou_C);
      p_dl += ((giou_C * dU_wrt_l) - (U * dC_wrt_l)) / (giou_C * giou_C);
      p_dr += ((giou_C * dU_wrt_r) - (U * dC_wrt_r)) / (giou_C * giou_C);
    }
    if (Iw <= 0 || Ih <= 0) {
      p_dt = ((giou_C * dU_wrt_t) - (U * dC_wrt_t)) / (giou_C * giou_C);
      p_db = ((giou_C * dU_wrt_b) - (U * dC_wrt_b)) / (giou_C * giou_C);
      p_dl = ((giou_C * dU_wrt_l) - (U * dC_wrt_l)) / (giou_C * giou_C);
      p_dr = ((giou_C * dU_wrt_r) - (U * dC_wrt_r)) / (giou_C * giou_C);
    }
  }

  float Ct = fmin(pred.y - pred.h / 2, truth.y - truth.h / 2);
  float Cb = fmax(pred.y + pred.h / 2, truth.y + truth.h / 2);
  float Cl = fmin(pred.x - pred.w / 2, truth.x - truth.w / 2);
  float Cr = fmax(pred.x + pred.w / 2, truth.x + truth.w / 2);
  float Cw = Cr - Cl;
  float Ch = Cb - Ct;
  float C = Cw * Cw + Ch * Ch;

  float dCt_dx = 0;
  float dCt_dy = pred_t < truth_tblr.top ? 1 : 0;
  float dCt_dw = 0;
  float dCt_dh = pred_t < truth_tblr.top ? -0.5 : 0;

  float dCb_dx = 0;
  float dCb_dy = pred_b > truth_tblr.bot ? 1 : 0;
  float dCb_dw = 0;
  float dCb_dh = pred_b > truth_tblr.bot ? 0.5 : 0;

  float dCl_dx = pred_l < truth_tblr.left ? 1 : 0;
  float dCl_dy = 0;
  float dCl_dw = pred_l < truth_tblr.left ? -0.5 : 0;
  float dCl_dh = 0;

  float dCr_dx = pred_r > truth_tblr.right ? 1 : 0;
  float dCr_dy = 0;
  float dCr_dw = pred_r > truth_tblr.right ? 0.5 : 0;
  float dCr_dh = 0;

  float dCw_dx = dCr_dx - dCl_dx;
  float dCw_dy = dCr_dy - dCl_dy;
  float dCw_dw = dCr_dw - dCl_dw;
  float dCw_dh = dCr_dh - dCl_dh;

  float dCh_dx = dCb_dx - dCt_dx;
  float dCh_dy = dCb_dy - dCt_dy;
  float dCh_dw = dCb_dw - dCt_dw;
  float dCh_dh = dCb_dh - dCt_dh;

  // UNUSED
  //// ground truth
  // float dI_wrt_xhat_t = pred_t < truth_tblr.top ? (-1 * Iw) : 0;
  // float dI_wrt_xhat_b = pred_b > truth_tblr.bot ? Iw : 0;
  // float dI_wrt_xhat_l = pred_l < truth_tblr.left ? (-1 * Ih) : 0;
  // float dI_wrt_xhat_r = pred_r > truth_tblr.right ? Ih : 0;

  // Final IOU loss (prediction) (negative of IOU gradient, we want the negative
  // loss)
  float p_dx = 0;
  float p_dy = 0;
  float p_dw = 0;
  float p_dh = 0;

  // p_dx, p_dy, p_dw and p_dh are the gradient of IoU or GIoU.
  p_dx = p_dl + p_dr;
  p_dy = p_dt + p_db;
  // For dw and dh, we do not divided by 2.
  p_dw = (p_dr - p_dl);
  p_dh = (p_db - p_dt);

  // https://github.com/Zzh-tju/DIoU-darknet
  // https://arxiv.org/abs/1911.08287
  if (iou_loss == DIOU) {
    if (C > 0) {
      p_dx += (2 * (truth.x - pred.x) * C -
               (2 * Cw * dCw_dx + 2 * Ch * dCh_dx) * S) /
              (C * C);
      p_dy += (2 * (truth.y - pred.y) * C -
               (2 * Cw * dCw_dy + 2 * Ch * dCh_dy) * S) /
              (C * C);
      p_dw += (2 * Cw * dCw_dw + 2 * Ch * dCh_dw) * S / (C * C);
      p_dh += (2 * Cw * dCw_dh + 2 * Ch * dCh_dh) * S / (C * C);
    }
    if (Iw <= 0 || Ih <= 0) {
      p_dx = (2 * (truth.x - pred.x) * C -
              (2 * Cw * dCw_dx + 2 * Ch * dCh_dx) * S) /
             (C * C);
      p_dy = (2 * (truth.y - pred.y) * C -
              (2 * Cw * dCw_dy + 2 * Ch * dCh_dy) * S) /
             (C * C);
      p_dw = (2 * Cw * dCw_dw + 2 * Ch * dCh_dw) * S / (C * C);
      p_dh = (2 * Cw * dCw_dh + 2 * Ch * dCh_dh) * S / (C * C);
    }
  }
  // The following codes are calculating the gradient of ciou.

  if (iou_loss == CIOU) {
    float ar_gt = truth.w / truth.h;
    float ar_pred = pred.w / pred.h;
    float ar_loss = 4 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) *
                    (atan(ar_gt) - atan(ar_pred));
    float alpha = ar_loss / (1 - I / U + ar_loss + 0.000001);
    float ar_dw = 8 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * pred.h;
    float ar_dh = -8 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * pred.w;
    if (C > 0) {
      // dar*
      p_dx += (2 * (truth.x - pred.x) * C -
               (2 * Cw * dCw_dx + 2 * Ch * dCh_dx) * S) /
              (C * C);
      p_dy += (2 * (truth.h - pred.h) * C -
               (2 * Cw * dCw_dy + 2 * Ch * dCh_dy) * S) /
              (C * C);
      p_dw += (2 * Cw * dCw_dw + 2 * Ch * dCh_dw) * S / (C * C) + alpha * ar_dw;
      p_dh += (2 * Cw * dCw_dh + 2 * Ch * dCh_dh) * S / (C * C) + alpha * ar_dh;
    }
    if (Iw <= 0 || Ih <= 0) {
      p_dx = (2 * (truth.x - pred.x) * C -
              (2 * Cw * dCw_dx + 2 * Ch * dCh_dx) * S) /
             (C * C);
      p_dy = (2 * (truth.y - pred.y) * C -
              (2 * Cw * dCw_dy + 2 * Ch * dCh_dy) * S) /
             (C * C);
      p_dw = (2 * Cw * dCw_dw + 2 * Ch * dCh_dw) * S / (C * C) + alpha * ar_dw;
      p_dh = (2 * Cw * dCw_dh + 2 * Ch * dCh_dh) * S / (C * C) + alpha * ar_dh;
    }
  }

  ddx.dt = p_dx; // We follow the original code released from GDarknet. So in
  // yolo_layer.c, dt, db, dl, dr are already dx, dy, dw, dh.
  ddx.db = p_dy;
  ddx.dl = p_dw;
  ddx.dr = p_dh;

  return ddx;
}
template <typename Dtype>
dxrep dx_box_iou(const vector<Dtype> &pred, const vector<Dtype> &truth,
                 IOU_LOSS iou_loss) {
  return dx_box_iou(box(pred), box(truth), iou_loss);
}

template <typename Dtype>
boxabs box_c(const vector<Dtype> &a, const vector<Dtype> &b) {
  boxabs ba = {0};
  ba.top = fmin(a[1] - a[3] / 2, b[1] - b[3] / 2);
  ba.bot = fmax(a[1] + a[3] / 2, b[1] + b[3] / 2);
  ba.left = fmin(a[0] - a[2] / 2, b[0] - b[2] / 2);
  ba.right = fmax(a[0] + a[2] / 2, b[0] + b[2] / 2);
  return ba;
}
template boxabs box_c(const vector<float> &a, const vector<float> &b);
template boxabs box_c(const vector<double> &a, const vector<double> &b);

boxabs box_c(const box &a, const box &b) {
  return box_c(a.to_vector<float>(), b.to_vector<float>());
}

// representation from x, y, w, h to top, left, bottom, right
template <typename Dtype>
boxabs to_tblr(const vector<Dtype> &a) {
  boxabs tblr = {0};
  float t = a[1] - (a[3] / 2);
  float b = a[1] + (a[3] / 2);
  float l = a[0] - (a[2] / 2);
  float r = a[0] + (a[2] / 2);
  tblr.top = t;
  tblr.bot = b;
  tblr.left = l;
  tblr.right = r;
  return tblr;
}
template boxabs to_tblr(const vector<float> &);
template boxabs to_tblr(const vector<double> &);
boxabs to_tblr(const box &a) { return to_tblr(a.to_vector<float>()); }

template <typename Dtype>
Dtype box_giou(const vector<Dtype> &a, const vector<Dtype> &b) {
  boxabs ba = box_c(a, b);
  float w = ba.right - ba.left;
  float h = ba.bot - ba.top;
  float c = w * h;
  float iou = box_iou(a, b);
  if (c == 0) {
    return iou;
  }
  float u = box_union(a, b);
  float giou_term = (c - u) / c;

  return iou - giou_term;
}
template float box_giou(const vector<float> &a, const vector<float> &b);
template double box_giou(const vector<double> &a, const vector<double> &b);

float box_giou(const box &a, const box &b) {
  return box_giou(a.to_vector<float>(), b.to_vector<float>());
}

// https://github.com/Zzh-tju/DIoU-darknet
// https://arxiv.org/abs/1911.08287
template <typename Dtype>
Dtype box_diou(const vector<Dtype> &a, const vector<Dtype> &b) {
  boxabs ba = box_c(a, b);
  Dtype w = ba.right - ba.left;
  Dtype h = ba.bot - ba.top;
  Dtype c = w * w + h * h;
  Dtype iou = box_iou(a, b);
  if (c == 0) {
    return iou;
  }
  Dtype d = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
  Dtype u = pow(d / c, 0.6);
  Dtype diou_term = u;

  return iou - diou_term;
}
template float box_diou(const vector<float> &a, const vector<float> &b);
template double box_diou(const vector<double> &a, const vector<double> &b);

float box_diou(const box &a, const box &b) {
  return box_giou(a.to_vector<float>(), b.to_vector<float>());
}

// https://github.com/Zzh-tju/DIoU-darknet
// https://arxiv.org/abs/1911.08287
template <typename Dtype>
Dtype box_ciou(const vector<Dtype> &a, const vector<Dtype> &b) {
  boxabs ba = box_c(a, b);
  Dtype w = ba.right - ba.left;
  Dtype h = ba.bot - ba.top;
  Dtype c = w * w + h * h;
  Dtype iou = box_iou(a, b);
  if (c == 0) {
    return iou;
  }
  Dtype u = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
  Dtype d = u / c;
  Dtype ar_gt = b[2] / b[3];
  Dtype ar_pred = a[2] / a[3];
  Dtype ar_loss = 4 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) *
                  (atan(ar_gt) - atan(ar_pred));
  Dtype alpha = ar_loss / (1 - iou + ar_loss + 0.000001);
  Dtype ciou_term = d + alpha * ar_loss; // ciou
  return iou - ciou_term;
}
template float box_ciou(const vector<float> &a, const vector<float> &b);
template double box_ciou(const vector<double> &a, const vector<double> &b);

float box_ciou(const box &a, const box &b) {
  return box_giou(a.to_vector<float>(), b.to_vector<float>());
}

float box_diounms(vector<float> a, vector<float> b, float beta1) {
  boxabs ba = box_c(a, b);
  float w = ba.right - ba.left;
  float h = ba.bot - ba.top;
  float c = w * w + h * h;
  float iou = box_iou(a, b);
  if (c == 0) {
    return iou;
  }
  float d = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
  float u = pow(d / c, beta1);
  float diou_term = u;
#ifdef DEBUG_PRINTS
  printf("  c: %f, u: %f, riou_term: %f\n", c, u, diou_term);
#endif
  return iou - diou_term;
}

bool SortBBoxAscend(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2) {
  return bbox1.score() < bbox2.score();
}

bool SortBBoxDescend(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2) {
  return bbox1.score() > bbox2.score();
}

template <typename T>
bool SortScorePairAscend(const pair<float, T> &pair1,
                         const pair<float, T> &pair2) {
  return pair1.first < pair2.first;
}

// Explicit initialization.
template bool SortScorePairAscend(const pair<float, int> &pair1,
                                  const pair<float, int> &pair2);
template bool SortScorePairAscend(const pair<float, pair<int, int>> &pair1,
                                  const pair<float, pair<int, int>> &pair2);

template <typename T>
bool SortScorePairDescend(const pair<float, T> &pair1,
                          const pair<float, T> &pair2) {
  return pair1.first > pair2.first;
}

// Explicit initialization.
template bool SortScorePairDescend(const pair<float, int> &pair1,
                                   const pair<float, int> &pair2);
template bool SortScorePairDescend(const pair<float, pair<int, int>> &pair1,
                                   const pair<float, pair<int, int>> &pair2);

NormalizedBBox UnitBBox() {
  NormalizedBBox unit_bbox;
  unit_bbox.set_xmin(0.);
  unit_bbox.set_ymin(0.);
  unit_bbox.set_xmax(1.);
  unit_bbox.set_ymax(1.);
  return unit_bbox;
}

bool IsCrossBoundaryBBox(const NormalizedBBox &bbox) {
  return bbox.xmin() < 0 || bbox.xmin() > 1 || bbox.ymin() < 0 ||
         bbox.ymin() > 1 || bbox.xmax() < 0 || bbox.xmax() > 1 ||
         bbox.ymax() < 0 || bbox.ymax() > 1;
}

void IntersectBBox(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2,
                   NormalizedBBox *intersect_bbox) {
  if (bbox2.xmin() > bbox1.xmax() || bbox2.xmax() < bbox1.xmin() ||
      bbox2.ymin() > bbox1.ymax() || bbox2.ymax() < bbox1.ymin()) {
    // Return [0, 0, 0, 0] if there is no intersection.
    intersect_bbox->set_xmin(0);
    intersect_bbox->set_ymin(0);
    intersect_bbox->set_xmax(0);
    intersect_bbox->set_ymax(0);
  } else {
    intersect_bbox->set_xmin(std::max(bbox1.xmin(), bbox2.xmin()));
    intersect_bbox->set_ymin(std::max(bbox1.ymin(), bbox2.ymin()));
    intersect_bbox->set_xmax(std::min(bbox1.xmax(), bbox2.xmax()));
    intersect_bbox->set_ymax(std::min(bbox1.ymax(), bbox2.ymax()));
  }
}

float BBoxSize(const NormalizedBBox &bbox, const bool normalized) {
  if (bbox.xmax() < bbox.xmin() || bbox.ymax() < bbox.ymin()) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return 0;
  } else {
    if (bbox.has_size()) {
      return bbox.size();
    } else {
      float width = bbox.xmax() - bbox.xmin();
      float height = bbox.ymax() - bbox.ymin();
      if (normalized) {
        return width * height;
      } else {
        // If bbox is not within range [0, 1].
        return (width + 1) * (height + 1);
      }
    }
  }
}

template <typename Dtype>
Dtype BBoxSize(const Dtype *bbox, const bool normalized) {
  if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return Dtype(0.);
  } else {
    const Dtype width = bbox[2] - bbox[0];
    const Dtype height = bbox[3] - bbox[1];
    if (normalized) {
      return width * height;
    } else {
      // If bbox is not within range [0, 1].
      return (width + 1) * (height + 1);
    }
  }
}

template float BBoxSize(const float *bbox, const bool normalized);
template double BBoxSize(const double *bbox, const bool normalized);

void ClipBBox(const NormalizedBBox &bbox, NormalizedBBox *clip_bbox) {
  clip_bbox->set_xmin(caffe_clip(bbox.xmin(), 0.f, 1.f));
  clip_bbox->set_ymin(caffe_clip(bbox.ymin(), 0.f, 1.f));
  clip_bbox->set_xmax(caffe_clip(bbox.xmax(), 0.f, 1.f));
  clip_bbox->set_ymax(caffe_clip(bbox.ymax(), 0.f, 1.f));
  clip_bbox->clear_size();
  clip_bbox->set_size(BBoxSize(*clip_bbox));
  clip_bbox->set_difficult(bbox.difficult());
}

void ClipBBox(const NormalizedBBox &bbox, const float height, const float width,
              NormalizedBBox *clip_bbox) {
  clip_bbox->set_xmin(caffe_clip(bbox.xmin(), 0.f, width));
  clip_bbox->set_ymin(caffe_clip(bbox.ymin(), 0.f, height));
  clip_bbox->set_xmax(caffe_clip(bbox.xmax(), 0.f, width));
  clip_bbox->set_ymax(caffe_clip(bbox.ymax(), 0.f, height));
  clip_bbox->clear_size();
  clip_bbox->set_size(BBoxSize(*clip_bbox));
  clip_bbox->set_difficult(bbox.difficult());
}

void ScaleBBox(const NormalizedBBox &bbox, const int height, const int width,
               NormalizedBBox *scale_bbox) {
  scale_bbox->set_xmin(bbox.xmin() * width);
  scale_bbox->set_ymin(bbox.ymin() * height);
  scale_bbox->set_xmax(bbox.xmax() * width);
  scale_bbox->set_ymax(bbox.ymax() * height);
  scale_bbox->clear_size();
  bool normalized = !(width > 1 || height > 1);
  scale_bbox->set_size(BBoxSize(*scale_bbox, normalized));
  scale_bbox->set_difficult(bbox.difficult());
}

void OutputBBox(const NormalizedBBox &bbox, const pair<int, int> &img_size,
                const bool has_resize, const ResizeParameter &resize_param,
                NormalizedBBox *out_bbox) {
  const int height = img_size.first;
  const int width = img_size.second;
  NormalizedBBox temp_bbox = bbox;
  if (has_resize && resize_param.resize_mode()) {
    float resize_height = resize_param.height();
    CHECK_GT(resize_height, 0);
    float resize_width = resize_param.width();
    CHECK_GT(resize_width, 0);
    float resize_aspect = resize_width / resize_height;
    int height_scale = resize_param.height_scale();
    int width_scale = resize_param.width_scale();
    float aspect = static_cast<float>(width) / height;

    float padding;
    NormalizedBBox source_bbox;
    switch (resize_param.resize_mode()) {
    case ResizeParameter_Resize_mode_WARP:
      ClipBBox(temp_bbox, &temp_bbox);
      ScaleBBox(temp_bbox, height, width, out_bbox);
      break;
    case ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD:
      if (aspect > resize_aspect) {
        padding = (resize_height - resize_width / aspect) / 2;
        source_bbox.set_xmin(0.);
        source_bbox.set_ymin(padding / resize_height);
        source_bbox.set_xmax(1.);
        source_bbox.set_ymax(1. - padding / resize_height);
      } else {
        padding = (resize_width - resize_height * aspect) / 2;
        source_bbox.set_xmin(padding / resize_width);
        source_bbox.set_ymin(0.);
        source_bbox.set_xmax(1. - padding / resize_width);
        source_bbox.set_ymax(1.);
      }
      ProjectBBox(source_bbox, bbox, &temp_bbox);
      ClipBBox(temp_bbox, &temp_bbox);
      ScaleBBox(temp_bbox, height, width, out_bbox);
      break;
    case ResizeParameter_Resize_mode_FIT_SMALL_SIZE:
      if (height_scale == 0 || width_scale == 0) {
        ClipBBox(temp_bbox, &temp_bbox);
        ScaleBBox(temp_bbox, height, width, out_bbox);
      } else {
        ScaleBBox(temp_bbox, height_scale, width_scale, out_bbox);
        ClipBBox(*out_bbox, height, width, out_bbox);
      }
      break;
    default:
      LOG(FATAL) << "Unknown resize mode.";
    }
  } else {
    // Clip the normalized bbox first.
    ClipBBox(temp_bbox, &temp_bbox);
    // Scale the bbox according to the original image size.
    ScaleBBox(temp_bbox, height, width, out_bbox);
  }
}

void LocateBBox(const NormalizedBBox &src_bbox, const NormalizedBBox &bbox,
                NormalizedBBox *loc_bbox) {
  float src_width = src_bbox.xmax() - src_bbox.xmin();
  float src_height = src_bbox.ymax() - src_bbox.ymin();
  loc_bbox->set_xmin(src_bbox.xmin() + bbox.xmin() * src_width);
  loc_bbox->set_ymin(src_bbox.ymin() + bbox.ymin() * src_height);
  loc_bbox->set_xmax(src_bbox.xmin() + bbox.xmax() * src_width);
  loc_bbox->set_ymax(src_bbox.ymin() + bbox.ymax() * src_height);
  loc_bbox->set_difficult(bbox.difficult());
}

bool ProjectBBox(const NormalizedBBox &src_bbox, const NormalizedBBox &bbox,
                 NormalizedBBox *proj_bbox) {
  if (bbox.xmin() >= src_bbox.xmax() || bbox.xmax() <= src_bbox.xmin() ||
      bbox.ymin() >= src_bbox.ymax() || bbox.ymax() <= src_bbox.ymin()) {
    return false;
  }
  float src_width = src_bbox.xmax() - src_bbox.xmin();
  float src_height = src_bbox.ymax() - src_bbox.ymin();
  proj_bbox->set_xmin((bbox.xmin() - src_bbox.xmin()) / src_width);
  proj_bbox->set_ymin((bbox.ymin() - src_bbox.ymin()) / src_height);
  proj_bbox->set_xmax((bbox.xmax() - src_bbox.xmin()) / src_width);
  proj_bbox->set_ymax((bbox.ymax() - src_bbox.ymin()) / src_height);
  proj_bbox->set_difficult(bbox.difficult());
  ClipBBox(*proj_bbox, proj_bbox);
  return BBoxSize(*proj_bbox) > 0;
}

void ExtrapolateBBox(const ResizeParameter &param, const int height,
                     const int width, const NormalizedBBox &crop_bbox,
                     NormalizedBBox *bbox) {
  float height_scale = param.height_scale();
  float width_scale = param.width_scale();
  if (height_scale > 0 && width_scale > 0 &&
      param.resize_mode() == ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
    float orig_aspect = static_cast<float>(width) / height;
    float resize_height = param.height();
    float resize_width = param.width();
    float resize_aspect = resize_width / resize_height;
    if (orig_aspect < resize_aspect) {
      resize_height = resize_width / orig_aspect;
    } else {
      resize_width = resize_height * orig_aspect;
    }
    float crop_height = resize_height * (crop_bbox.ymax() - crop_bbox.ymin());
    float crop_width = resize_width * (crop_bbox.xmax() - crop_bbox.xmin());
    CHECK_GE(crop_width, width_scale);
    CHECK_GE(crop_height, height_scale);
    bbox->set_xmin(bbox->xmin() * crop_width / width_scale);
    bbox->set_xmax(bbox->xmax() * crop_width / width_scale);
    bbox->set_ymin(bbox->ymin() * crop_height / height_scale);
    bbox->set_ymax(bbox->ymax() * crop_height / height_scale);
  }
}

float JaccardOverlap(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2,
                     const bool normalized) {
  NormalizedBBox intersect_bbox;
  IntersectBBox(bbox1, bbox2, &intersect_bbox);
  float intersect_width, intersect_height;
  if (normalized) {
    intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin();
    intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin();
  } else {
    intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin() + 1;
    intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin() + 1;
  }
  if (intersect_width > 0 && intersect_height > 0) {
    float intersect_size = intersect_width * intersect_height;
    float bbox1_size = BBoxSize(bbox1);
    float bbox2_size = BBoxSize(bbox2);
    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
  } else {
    return 0.;
  }
}

template <typename Dtype>
Dtype JaccardOverlap(const Dtype *bbox1, const Dtype *bbox2) {
  if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] || bbox2[1] > bbox1[3] ||
      bbox2[3] < bbox1[1]) {
    return Dtype(0.);
  } else {
    const Dtype inter_xmin = std::max(bbox1[0], bbox2[0]);
    const Dtype inter_ymin = std::max(bbox1[1], bbox2[1]);
    const Dtype inter_xmax = std::min(bbox1[2], bbox2[2]);
    const Dtype inter_ymax = std::min(bbox1[3], bbox2[3]);

    const Dtype inter_width = inter_xmax - inter_xmin;
    const Dtype inter_height = inter_ymax - inter_ymin;
    const Dtype inter_size = inter_width * inter_height;

    const Dtype bbox1_size = BBoxSize(bbox1);
    const Dtype bbox2_size = BBoxSize(bbox2);

    return inter_size / (bbox1_size + bbox2_size - inter_size);
  }
}

template float JaccardOverlap(const float *bbox1, const float *bbox2);
template double JaccardOverlap(const double *bbox1, const double *bbox2);

float BBoxCoverage(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2) {
  NormalizedBBox intersect_bbox;
  IntersectBBox(bbox1, bbox2, &intersect_bbox);
  float intersect_size = BBoxSize(intersect_bbox);
  if (intersect_size > 0) {
    float bbox1_size = BBoxSize(bbox1);
    return intersect_size / bbox1_size;
  } else {
    return 0.;
  }
}

bool MeetEmitConstraint(const NormalizedBBox &src_bbox,
                        const NormalizedBBox &bbox,
                        const EmitConstraint &emit_constraint) {
  EmitType emit_type = emit_constraint.emit_type();
  if (emit_type == EmitConstraint_EmitType_CENTER) {
    float x_center = (bbox.xmin() + bbox.xmax()) / 2;
    float y_center = (bbox.ymin() + bbox.ymax()) / 2;
    return x_center >= src_bbox.xmin() && x_center <= src_bbox.xmax() &&
           y_center >= src_bbox.ymin() && y_center <= src_bbox.ymax();
  } else if (emit_type == EmitConstraint_EmitType_MIN_OVERLAP) {
    float bbox_coverage = BBoxCoverage(bbox, src_bbox);
    return bbox_coverage > emit_constraint.emit_overlap();
  } else {
    LOG(FATAL) << "Unknown emit type.";
    return false;
  }
}

void EncodeBBox(const NormalizedBBox &prior_bbox,
                const vector<float> &prior_variance, const CodeType code_type,
                const bool encode_variance_in_target,
                const NormalizedBBox &bbox, NormalizedBBox *encode_bbox) {
  if (code_type == PriorBoxParameter_CodeType_CORNER) {
    if (encode_variance_in_target) {
      encode_bbox->set_xmin(bbox.xmin() - prior_bbox.xmin());
      encode_bbox->set_ymin(bbox.ymin() - prior_bbox.ymin());
      encode_bbox->set_xmax(bbox.xmax() - prior_bbox.xmax());
      encode_bbox->set_ymax(bbox.ymax() - prior_bbox.ymax());
    } else {
      // Encode variance in bbox.
      CHECK_EQ(prior_variance.size(), 4);
      for (float i : prior_variance) {
        CHECK_GT(i, 0);
      }
      encode_bbox->set_xmin((bbox.xmin() - prior_bbox.xmin()) /
                            prior_variance[0]);
      encode_bbox->set_ymin((bbox.ymin() - prior_bbox.ymin()) /
                            prior_variance[1]);
      encode_bbox->set_xmax((bbox.xmax() - prior_bbox.xmax()) /
                            prior_variance[2]);
      encode_bbox->set_ymax((bbox.ymax() - prior_bbox.ymax()) /
                            prior_variance[3]);
    }
  } else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
    float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
    CHECK_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
    CHECK_GT(prior_height, 0);
    float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.F;
    float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.F;

    float bbox_width = bbox.xmax() - bbox.xmin();
    CHECK_GT(bbox_width, 0);
    float bbox_height = bbox.ymax() - bbox.ymin();
    CHECK_GT(bbox_height, 0);
    float bbox_center_x = (bbox.xmin() + bbox.xmax()) / 2.F;
    float bbox_center_y = (bbox.ymin() + bbox.ymax()) / 2.F;

    if (encode_variance_in_target) {
      encode_bbox->set_xmin((bbox_center_x - prior_center_x) / prior_width);
      encode_bbox->set_ymin((bbox_center_y - prior_center_y) / prior_height);
      encode_bbox->set_xmax(log(bbox_width / prior_width));
      encode_bbox->set_ymax(log(bbox_height / prior_height));
    } else {
      // Encode variance in bbox.
      encode_bbox->set_xmin((bbox_center_x - prior_center_x) / prior_width /
                            prior_variance[0]);
      encode_bbox->set_ymin((bbox_center_y - prior_center_y) / prior_height /
                            prior_variance[1]);
      encode_bbox->set_xmax(log(bbox_width / prior_width) / prior_variance[2]);
      encode_bbox->set_ymax(log(bbox_height / prior_height) /
                            prior_variance[3]);
    }
  } else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
    float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
    CHECK_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
    CHECK_GT(prior_height, 0);
    if (encode_variance_in_target) {
      encode_bbox->set_xmin((bbox.xmin() - prior_bbox.xmin()) / prior_width);
      encode_bbox->set_ymin((bbox.ymin() - prior_bbox.ymin()) / prior_height);
      encode_bbox->set_xmax((bbox.xmax() - prior_bbox.xmax()) / prior_width);
      encode_bbox->set_ymax((bbox.ymax() - prior_bbox.ymax()) / prior_height);
    } else {
      // Encode variance in bbox.
      CHECK_EQ(prior_variance.size(), 4);
      for (float i : prior_variance) {
        CHECK_GT(i, 0);
      }
      encode_bbox->set_xmin((bbox.xmin() - prior_bbox.xmin()) / prior_width /
                            prior_variance[0]);
      encode_bbox->set_ymin((bbox.ymin() - prior_bbox.ymin()) / prior_height /
                            prior_variance[1]);
      encode_bbox->set_xmax((bbox.xmax() - prior_bbox.xmax()) / prior_width /
                            prior_variance[2]);
      encode_bbox->set_ymax((bbox.ymax() - prior_bbox.ymax()) / prior_height /
                            prior_variance[3]);
    }
  } else {
    LOG(FATAL) << "Unknown LocLossType.";
  }
}

void DecodeBBox(const NormalizedBBox &prior_bbox,
                const vector<float> &prior_variance, const CodeType code_type,
                const bool variance_encoded_in_target, const bool clip_bbox,
                const NormalizedBBox &bbox, NormalizedBBox *decode_bbox) {
  if (code_type == PriorBoxParameter_CodeType_CORNER) {
    if (variance_encoded_in_target) {
      // variance is encoded in target, we simply need to add the offset
      // predictions.
      decode_bbox->set_xmin(prior_bbox.xmin() + bbox.xmin());
      decode_bbox->set_ymin(prior_bbox.ymin() + bbox.ymin());
      decode_bbox->set_xmax(prior_bbox.xmax() + bbox.xmax());
      decode_bbox->set_ymax(prior_bbox.ymax() + bbox.ymax());
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox->set_xmin(prior_bbox.xmin() +
                            prior_variance[0] * bbox.xmin());
      decode_bbox->set_ymin(prior_bbox.ymin() +
                            prior_variance[1] * bbox.ymin());
      decode_bbox->set_xmax(prior_bbox.xmax() +
                            prior_variance[2] * bbox.xmax());
      decode_bbox->set_ymax(prior_bbox.ymax() +
                            prior_variance[3] * bbox.ymax());
    }
  } else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
    float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
    CHECK_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
    CHECK_GT(prior_height, 0);
    float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.F;
    float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.F;

    float decode_bbox_center_x, decode_bbox_center_y;
    float decode_bbox_width, decode_bbox_height;
    if (variance_encoded_in_target) {
      // variance is encoded in target, we simply need to retore the offset
      // predictions.
      decode_bbox_center_x = bbox.xmin() * prior_width + prior_center_x;
      decode_bbox_center_y = bbox.ymin() * prior_height + prior_center_y;
      decode_bbox_width = exp(bbox.xmax()) * prior_width;
      decode_bbox_height = exp(bbox.ymax()) * prior_height;
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox_center_x =
          prior_variance[0] * bbox.xmin() * prior_width + prior_center_x;
      decode_bbox_center_y =
          prior_variance[1] * bbox.ymin() * prior_height + prior_center_y;
      decode_bbox_width = exp(prior_variance[2] * bbox.xmax()) * prior_width;
      decode_bbox_height = exp(prior_variance[3] * bbox.ymax()) * prior_height;
    }

    decode_bbox->set_xmin(decode_bbox_center_x - decode_bbox_width / 2.F);
    decode_bbox->set_ymin(decode_bbox_center_y - decode_bbox_height / 2.F);
    decode_bbox->set_xmax(decode_bbox_center_x + decode_bbox_width / 2.F);
    decode_bbox->set_ymax(decode_bbox_center_y + decode_bbox_height / 2.F);
  } else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
    float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
    CHECK_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
    CHECK_GT(prior_height, 0);
    if (variance_encoded_in_target) {
      // variance is encoded in target, we simply need to add the offset
      // predictions.
      decode_bbox->set_xmin(prior_bbox.xmin() + bbox.xmin() * prior_width);
      decode_bbox->set_ymin(prior_bbox.ymin() + bbox.ymin() * prior_height);
      decode_bbox->set_xmax(prior_bbox.xmax() + bbox.xmax() * prior_width);
      decode_bbox->set_ymax(prior_bbox.ymax() + bbox.ymax() * prior_height);
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox->set_xmin(prior_bbox.xmin() +
                            prior_variance[0] * bbox.xmin() * prior_width);
      decode_bbox->set_ymin(prior_bbox.ymin() +
                            prior_variance[1] * bbox.ymin() * prior_height);
      decode_bbox->set_xmax(prior_bbox.xmax() +
                            prior_variance[2] * bbox.xmax() * prior_width);
      decode_bbox->set_ymax(prior_bbox.ymax() +
                            prior_variance[3] * bbox.ymax() * prior_height);
    }
  } else {
    LOG(FATAL) << "Unknown LocLossType.";
  }
  float bbox_size = BBoxSize(*decode_bbox);
  decode_bbox->set_size(bbox_size);
  if (clip_bbox) {
    ClipBBox(*decode_bbox, decode_bbox);
  }
}

void DecodeBBoxes(const vector<NormalizedBBox> &prior_bboxes,
                  const vector<vector<float>> &prior_variances,
                  const CodeType code_type,
                  const bool variance_encoded_in_target, const bool clip_bbox,
                  const vector<NormalizedBBox> &bboxes,
                  vector<NormalizedBBox> *decode_bboxes) {
  CHECK_EQ(prior_bboxes.size(), prior_variances.size());
  CHECK_EQ(prior_bboxes.size(), bboxes.size());
  int num_bboxes = prior_bboxes.size();
  if (num_bboxes >= 1) {
    CHECK_EQ(prior_variances[0].size(), 4);
  }
  decode_bboxes->clear();
  for (int i = 0; i < num_bboxes; ++i) {
    NormalizedBBox decode_bbox;
    DecodeBBox(prior_bboxes[i], prior_variances[i], code_type,
               variance_encoded_in_target, clip_bbox, bboxes[i], &decode_bbox);
    decode_bboxes->push_back(decode_bbox);
  }
}

void DecodeBBoxesAll(const vector<LabelBBox> &all_loc_preds,
                     const vector<NormalizedBBox> &prior_bboxes,
                     const vector<vector<float>> &prior_variances,
                     const int num, const bool share_location,
                     const int num_loc_classes, const int background_label_id,
                     const CodeType code_type,
                     const bool variance_encoded_in_target, const bool clip,
                     vector<LabelBBox> *all_decode_bboxes) {
  CHECK_EQ(all_loc_preds.size(), num);
  all_decode_bboxes->clear();
  all_decode_bboxes->resize(num);
  for (int i = 0; i < num; ++i) {
    // Decode predictions into bboxes.
    LabelBBox &decode_bboxes = (*all_decode_bboxes)[i];
    for (int c = 0; c < num_loc_classes; ++c) {
      int label = share_location ? -1 : c;
      if (label == background_label_id) {
        // Ignore background class.
        continue;
      }
      if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for label " << label;
      }
      const vector<NormalizedBBox> &label_loc_preds =
          all_loc_preds[i].find(label)->second;
      DecodeBBoxes(prior_bboxes, prior_variances, code_type,
                   variance_encoded_in_target, clip, label_loc_preds,
                   &(decode_bboxes[label]));
    }
  }
}

void MatchBBox(const vector<NormalizedBBox> &gt_bboxes,
               const vector<NormalizedBBox> &pred_bboxes, const int label,
               const MatchType match_type, const float overlap_threshold,
               const bool ignore_cross_boundary_bbox,
               vector<int> *match_indices, vector<float> *match_overlaps) {
  int num_pred = pred_bboxes.size();
  match_indices->clear();
  match_indices->resize(num_pred, -1);
  match_overlaps->clear();
  match_overlaps->resize(num_pred, 0.);

  int num_gt = 0;
  vector<int> gt_indices;
  if (label == -1) {
    // label -1 means comparing against all ground truth.
    num_gt = gt_bboxes.size();
    for (int i = 0; i < num_gt; ++i) {
      gt_indices.push_back(i);
    }
  } else {
    // Count number of ground truth boxes which has the desired label.
    for (int i = 0; i < gt_bboxes.size(); ++i) {
      if (gt_bboxes[i].label() == label) {
        num_gt++;
        gt_indices.push_back(i);
      }
    }
  }
  if (num_gt == 0) {
    return;
  }

  // Store the positive overlap between predictions and ground truth.
  map<int, map<int, float>> overlaps;
  for (int i = 0; i < num_pred; ++i) {
    if (ignore_cross_boundary_bbox && IsCrossBoundaryBBox(pred_bboxes[i])) {
      (*match_indices)[i] = -2;
      continue;
    }
    for (int j = 0; j < num_gt; ++j) {
      float overlap = JaccardOverlap(pred_bboxes[i], gt_bboxes[gt_indices[j]]);
      if (overlap > 1e-6) {
        (*match_overlaps)[i] = std::max((*match_overlaps)[i], overlap);
        overlaps[i][j] = overlap;
      }
    }
  }

  // Bipartite matching.
  vector<int> gt_pool;
  for (int i = 0; i < num_gt; ++i) {
    gt_pool.push_back(i);
  }
  while (!gt_pool.empty()) {
    // Find the most overlapped gt and cooresponding predictions.
    int max_idx = -1;
    int max_gt_idx = -1;
    float max_overlap = -1;
    for (auto &overlap : overlaps) {
      int i = overlap.first;
      if ((*match_indices)[i] != -1) {
        // The prediction already has matched ground truth or is ignored.
        continue;
      }
      for (int j : gt_pool) {
        if (overlap.second.find(j) == overlap.second.end()) {
          // No overlap between the i-th prediction and j-th ground truth.
          continue;
        }
        // Find the maximum overlapped pair.
        if (overlap.second[j] > max_overlap) {
          // If the prediction has not been matched to any ground truth,
          // and the overlap is larger than maximum overlap, update.
          max_idx = i;
          max_gt_idx = j;
          max_overlap = overlap.second[j];
        }
      }
    }
    if (max_idx == -1) {
      // Cannot find good match.
      break;
    } else {
      CHECK_EQ((*match_indices)[max_idx], -1);
      (*match_indices)[max_idx] = gt_indices[max_gt_idx];
      (*match_overlaps)[max_idx] = max_overlap;
      // Erase the ground truth.
      gt_pool.erase(std::find(gt_pool.begin(), gt_pool.end(), max_gt_idx));
    }
  }

  switch (match_type) {
  case MultiBoxLossParameter_MatchType_BIPARTITE:
    // Already done.
    break;
  case MultiBoxLossParameter_MatchType_PER_PREDICTION:
    // Get most overlaped for the rest prediction bboxes.
    for (auto &it : overlaps) {
      int i = it.first;
      if ((*match_indices)[i] != -1) {
        // The prediction already has matched ground truth or is ignored.
        continue;
      }
      int max_gt_idx = -1;
      float max_overlap = -1;
      for (int j = 0; j < num_gt; ++j) {
        if (it.second.find(j) == it.second.end()) {
          // No overlap between the i-th prediction and j-th ground truth.
          continue;
        }
        // Find the maximum overlapped pair.
        float overlap = it.second[j];
        if (overlap >= overlap_threshold && overlap > max_overlap) {
          // If the prediction has not been matched to any ground truth,
          // and the overlap is larger than maximum overlap, update.
          max_gt_idx = j;
          max_overlap = overlap;
        }
      }
      if (max_gt_idx != -1) {
        // Found a matched ground truth.
        CHECK_EQ((*match_indices)[i], -1);
        (*match_indices)[i] = gt_indices[max_gt_idx];
        (*match_overlaps)[i] = max_overlap;
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown matching type.";
    break;
  }
}

void FindMatches(const vector<LabelBBox> &all_loc_preds,
                 const map<int, vector<NormalizedBBox>> &all_gt_bboxes,
                 const vector<NormalizedBBox> &prior_bboxes,
                 const vector<vector<float>> &prior_variances,
                 const MultiBoxLossParameter &multibox_loss_param,
                 vector<map<int, vector<float>>> *all_match_overlaps,
                 vector<map<int, vector<int>>> *all_match_indices) {
  // all_match_overlaps->clear();
  // all_match_indices->clear();
  // Get parameters.
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  const int num_classes = multibox_loss_param.num_classes();
  CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
  const bool share_location = multibox_loss_param.share_location();
  const int loc_classes = share_location ? 1 : num_classes;
  const MatchType match_type = multibox_loss_param.match_type();
  const float overlap_threshold = multibox_loss_param.overlap_threshold();
  const bool use_prior_for_matching =
      multibox_loss_param.use_prior_for_matching();
  const int background_label_id = multibox_loss_param.background_label_id();
  const CodeType code_type = multibox_loss_param.code_type();
  const bool encode_variance_in_target =
      multibox_loss_param.encode_variance_in_target();
  const bool ignore_cross_boundary_bbox =
      multibox_loss_param.ignore_cross_boundary_bbox();
  // Find the matches.
  int num = all_loc_preds.size();
  for (int i = 0; i < num; ++i) {
    map<int, vector<int>> match_indices;
    map<int, vector<float>> match_overlaps;
    // Check if there is ground truth for current image.
    if (all_gt_bboxes.find(i) == all_gt_bboxes.end()) {
      // There is no gt for current image. All predictions are negative.
      all_match_indices->push_back(match_indices);
      all_match_overlaps->push_back(match_overlaps);
      continue;
    }
    // Find match between predictions and ground truth.
    const vector<NormalizedBBox> &gt_bboxes = all_gt_bboxes.find(i)->second;
    if (!use_prior_for_matching) {
      for (int c = 0; c < loc_classes; ++c) {
        int label = share_location ? -1 : c;
        if (!share_location && label == background_label_id) {
          // Ignore background loc predictions.
          continue;
        }
        // Decode the prediction into bbox first.
        vector<NormalizedBBox> loc_bboxes;
        bool clip_bbox = false;
        DecodeBBoxes(prior_bboxes, prior_variances, code_type,
                     encode_variance_in_target, clip_bbox,
                     all_loc_preds[i].find(label)->second, &loc_bboxes);
        MatchBBox(gt_bboxes, loc_bboxes, label, match_type, overlap_threshold,
                  ignore_cross_boundary_bbox, &match_indices[label],
                  &match_overlaps[label]);
      }
    } else {
      // Use prior bboxes to match against all ground truth.
      vector<int> temp_match_indices;
      vector<float> temp_match_overlaps;
      const int label = -1;
      MatchBBox(gt_bboxes, prior_bboxes, label, match_type, overlap_threshold,
                ignore_cross_boundary_bbox, &temp_match_indices,
                &temp_match_overlaps);
      if (share_location) {
        match_indices[label] = temp_match_indices;
        match_overlaps[label] = temp_match_overlaps;
      } else {
        // Get ground truth label for each ground truth bbox.
        vector<int> gt_labels;
        for (const auto &gt_bboxe : gt_bboxes) {
          gt_labels.push_back(gt_bboxe.label());
        }
        // Distribute the matching results to different loc_class.
        for (int c = 0; c < loc_classes; ++c) {
          if (c == background_label_id) {
            // Ignore background loc predictions.
            continue;
          }
          match_indices[c].resize(temp_match_indices.size(), -1);
          match_overlaps[c] = temp_match_overlaps;
          for (int m = 0; m < temp_match_indices.size(); ++m) {
            if (temp_match_indices[m] > -1) {
              const int gt_idx = temp_match_indices[m];
              CHECK_LT(gt_idx, gt_labels.size());
              if (c == gt_labels[gt_idx]) {
                match_indices[c][m] = gt_idx;
              }
            }
          }
        }
      }
    }
    all_match_indices->push_back(match_indices);
    all_match_overlaps->push_back(match_overlaps);
  }
}

int CountNumMatches(const vector<map<int, vector<int>>> &all_match_indices,
                    const int num) {
  int num_matches = 0;
  for (int i = 0; i < num; ++i) {
    const map<int, vector<int>> &match_indices = all_match_indices[i];
    for (const auto &match_indice : match_indices) {
      const vector<int> &match_index = match_indice.second;
      for (int m : match_index) {
        if (m > -1) {
          ++num_matches;
        }
      }
    }
  }
  return num_matches;
}

inline bool IsEligibleMining(const MiningType mining_type, const int match_idx,
                             const float match_overlap,
                             const float neg_overlap) {
  if (mining_type == MultiBoxLossParameter_MiningType_MAX_NEGATIVE) {
    return match_idx == -1 && match_overlap < neg_overlap;
  } else {
    return mining_type == MultiBoxLossParameter_MiningType_HARD_EXAMPLE;
  }
}

template <typename Dtype>
void MineHardExamples(const Blob<Dtype> &conf_blob,
                      const vector<LabelBBox> &all_loc_preds,
                      const map<int, vector<NormalizedBBox>> &all_gt_bboxes,
                      const vector<NormalizedBBox> &prior_bboxes,
                      const vector<vector<float>> &prior_variances,
                      const vector<map<int, vector<float>>> &all_match_overlaps,
                      const MultiBoxLossParameter &multibox_loss_param,
                      int *num_matches, int *num_negs,
                      vector<map<int, vector<int>>> *all_match_indices,
                      vector<vector<int>> *all_neg_indices) {
  int num = all_loc_preds.size();
  // CHECK_EQ(num, all_match_overlaps.size());
  // CHECK_EQ(num, all_match_indices->size());
  // all_neg_indices->clear();
  *num_matches = CountNumMatches(*all_match_indices, num);
  *num_negs = 0;
  int num_priors = prior_bboxes.size();
  CHECK_EQ(num_priors, prior_variances.size());
  // Get parameters.
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  const int num_classes = multibox_loss_param.num_classes();
  CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
  const int background_label_id = multibox_loss_param.background_label_id();
  const bool use_prior_for_nms = multibox_loss_param.use_prior_for_nms();
  const ConfLossType conf_loss_type = multibox_loss_param.conf_loss_type();
  const MiningType mining_type = multibox_loss_param.mining_type();
  if (mining_type == MultiBoxLossParameter_MiningType_NONE) {
    return;
  }
  const LocLossType loc_loss_type = multibox_loss_param.loc_loss_type();
  const float neg_pos_ratio = multibox_loss_param.neg_pos_ratio();
  const float neg_overlap = multibox_loss_param.neg_overlap();
  const CodeType code_type = multibox_loss_param.code_type();
  const bool encode_variance_in_target =
      multibox_loss_param.encode_variance_in_target();
  const bool has_nms_param = multibox_loss_param.has_nms_param();
  float nms_threshold = 0;
  int top_k = -1;
  if (has_nms_param) {
    nms_threshold = multibox_loss_param.nms_param().nms_threshold();
    top_k = multibox_loss_param.nms_param().top_k();
  }
  const int sample_size = multibox_loss_param.sample_size();
  // Compute confidence losses based on matching results.
  vector<vector<float>> all_conf_loss;
#ifdef CPU_ONLY
  ComputeConfLoss(conf_blob.cpu_data(), num, num_priors, num_classes,
                  background_label_id, conf_loss_type, *all_match_indices,
                  all_gt_bboxes, &all_conf_loss);
#else
  ComputeConfLossGPU(conf_blob, num, num_priors, num_classes,
                     background_label_id, conf_loss_type, *all_match_indices,
                     all_gt_bboxes, &all_conf_loss);
#endif
  vector<vector<float>> all_loc_loss;
  if (mining_type == MultiBoxLossParameter_MiningType_HARD_EXAMPLE) {
    // Compute localization losses based on matching results.
    Blob<Dtype> loc_pred, loc_gt;
    if (*num_matches != 0) {
      vector<int> loc_shape(2, 1);
      loc_shape[1] = *num_matches * 4;
      loc_pred.Reshape(loc_shape);
      loc_gt.Reshape(loc_shape);
      Dtype *loc_pred_data = loc_pred.mutable_cpu_data();
      Dtype *loc_gt_data = loc_gt.mutable_cpu_data();
      EncodeLocPrediction(all_loc_preds, all_gt_bboxes, *all_match_indices,
                          prior_bboxes, prior_variances, multibox_loss_param,
                          loc_pred_data, loc_gt_data);
    }
    ComputeLocLoss(loc_pred, loc_gt, *all_match_indices, num, num_priors,
                   loc_loss_type, &all_loc_loss);
  } else {
    // No localization loss.
    for (int i = 0; i < num; ++i) {
      vector<float> loc_loss(num_priors, 0.f);
      all_loc_loss.push_back(loc_loss);
    }
  }
  for (int i = 0; i < num; ++i) {
    map<int, vector<int>> &match_indices = (*all_match_indices)[i];
    const map<int, vector<float>> &match_overlaps = all_match_overlaps[i];
    // loc + conf loss.
    const vector<float> &conf_loss = all_conf_loss[i];
    const vector<float> &loc_loss = all_loc_loss[i];
    vector<float> loss;
    std::transform(conf_loss.begin(), conf_loss.end(), loc_loss.begin(),
                   std::back_inserter(loss), std::plus<float>());
    // Pick negatives or hard examples based on loss.
    set<int> sel_indices;
    vector<int> neg_indices;
    for (auto it = match_indices.begin(); it != match_indices.end(); ++it) {
      const int label = it->first;
      int num_sel = 0;
      // Get potential indices and loss pairs.
      vector<pair<float, int>> loss_indices;
      for (int m = 0; m < match_indices[label].size(); ++m) {
        if (IsEligibleMining(mining_type, match_indices[label][m],
                             match_overlaps.find(label)->second[m],
                             neg_overlap)) {
          loss_indices.emplace_back(loss[m], m);
          ++num_sel;
        }
      }
      if (mining_type == MultiBoxLossParameter_MiningType_MAX_NEGATIVE) {
        int num_pos = 0;
        for (int m : match_indices[label]) {
          if (m > -1) {
            ++num_pos;
          }
        }
        num_sel = std::min(static_cast<int>(num_pos * neg_pos_ratio), num_sel);
      } else if (mining_type == MultiBoxLossParameter_MiningType_HARD_EXAMPLE) {
        CHECK_GT(sample_size, 0);
        num_sel = std::min(sample_size, num_sel);
      }
      // Select samples.
      if (has_nms_param && nms_threshold > 0) {
        // Do nms before selecting samples.
        vector<float> sel_loss;
        vector<NormalizedBBox> sel_bboxes;
        if (use_prior_for_nms) {
          for (int m = 0; m < match_indices[label].size(); ++m) {
            if (IsEligibleMining(mining_type, match_indices[label][m],
                                 match_overlaps.find(label)->second[m],
                                 neg_overlap)) {
              sel_loss.push_back(loss[m]);
              sel_bboxes.push_back(prior_bboxes[m]);
            }
          }
        } else {
          // Decode the prediction into bbox first.
          vector<NormalizedBBox> loc_bboxes;
          bool clip_bbox = false;
          DecodeBBoxes(prior_bboxes, prior_variances, code_type,
                       encode_variance_in_target, clip_bbox,
                       all_loc_preds[i].find(label)->second, &loc_bboxes);
          for (int m = 0; m < match_indices[label].size(); ++m) {
            if (IsEligibleMining(mining_type, match_indices[label][m],
                                 match_overlaps.find(label)->second[m],
                                 neg_overlap)) {
              sel_loss.push_back(loss[m]);
              sel_bboxes.push_back(loc_bboxes[m]);
            }
          }
        }
        // Do non-maximum suppression based on the loss.
        vector<int> nms_indices;
        ApplyNMS(sel_bboxes, sel_loss, nms_threshold, top_k, &nms_indices);
        if (nms_indices.size() < num_sel) {
          LOG(INFO) << "not enough sample after nms: " << nms_indices.size();
        }
        // Pick top example indices after nms.
        num_sel = std::min(static_cast<int>(nms_indices.size()), num_sel);
        for (int n = 0; n < num_sel; ++n) {
          sel_indices.insert(loss_indices[nms_indices[n]].second);
        }
      } else {
        // Pick top example indices based on loss.
        std::sort(loss_indices.begin(), loss_indices.end(),
                  SortScorePairDescend<int>);
        for (int n = 0; n < num_sel; ++n) {
          sel_indices.insert(loss_indices[n].second);
        }
      }
      // Update the match_indices and select neg_indices.
      for (int m = 0; m < match_indices[label].size(); ++m) {
        if (match_indices[label][m] > -1) {
          if (mining_type == MultiBoxLossParameter_MiningType_HARD_EXAMPLE &&
              sel_indices.find(m) == sel_indices.end()) {
            match_indices[label][m] = -1;
            *num_matches -= 1;
          }
        } else if (match_indices[label][m] == -1) {
          if (sel_indices.find(m) != sel_indices.end()) {
            neg_indices.push_back(m);
            *num_negs += 1;
          }
        }
      }
    }
    all_neg_indices->push_back(neg_indices);
  }
}

// Explicite initialization.
template void MineHardExamples(
    const Blob<float> &conf_blob, const vector<LabelBBox> &all_loc_preds,
    const map<int, vector<NormalizedBBox>> &all_gt_bboxes,
    const vector<NormalizedBBox> &prior_bboxes,
    const vector<vector<float>> &prior_variances,
    const vector<map<int, vector<float>>> &all_match_overlaps,
    const MultiBoxLossParameter &multibox_loss_param, int *num_matches,
    int *num_negs, vector<map<int, vector<int>>> *all_match_indices,
    vector<vector<int>> *all_neg_indices);
template void MineHardExamples(
    const Blob<double> &conf_blob, const vector<LabelBBox> &all_loc_preds,
    const map<int, vector<NormalizedBBox>> &all_gt_bboxes,
    const vector<NormalizedBBox> &prior_bboxes,
    const vector<vector<float>> &prior_variances,
    const vector<map<int, vector<float>>> &all_match_overlaps,
    const MultiBoxLossParameter &multibox_loss_param, int *num_matches,
    int *num_negs, vector<map<int, vector<int>>> *all_match_indices,
    vector<vector<int>> *all_neg_indices);

template <typename Dtype>
void GetGroundTruth(const Dtype *gt_data, const int num_gt,
                    const int background_label_id, const bool use_difficult_gt,
                    map<int, vector<NormalizedBBox>> *all_gt_bboxes) {
  all_gt_bboxes->clear();
  for (int i = 0; i < num_gt; ++i) {
    int start_idx = i * 8;
    int item_id = gt_data[start_idx];
    if (item_id == -1) {
      continue;
    }
    int label = gt_data[start_idx + 1];
    CHECK_NE(background_label_id, label)
        << "Found background label in the dataset.";
    bool difficult = static_cast<bool>(gt_data[start_idx + 7]);
    if (!use_difficult_gt && difficult) {
      // Skip reading difficult ground truth.
      continue;
    }
    NormalizedBBox bbox;
    bbox.set_label(label);
    bbox.set_xmin(gt_data[start_idx + 3]);
    bbox.set_ymin(gt_data[start_idx + 4]);
    bbox.set_xmax(gt_data[start_idx + 5]);
    bbox.set_ymax(gt_data[start_idx + 6]);
    bbox.set_difficult(difficult);
    float bbox_size = BBoxSize(bbox);
    bbox.set_size(bbox_size);
    (*all_gt_bboxes)[item_id].push_back(bbox);
  }
}

// Explicit initialization.
template void GetGroundTruth(const float *gt_data, const int num_gt,
                             const int background_label_id,
                             const bool use_difficult_gt,
                             map<int, vector<NormalizedBBox>> *all_gt_bboxes);
template void GetGroundTruth(const double *gt_data, const int num_gt,
                             const int background_label_id,
                             const bool use_difficult_gt,
                             map<int, vector<NormalizedBBox>> *all_gt_bboxes);

template <typename Dtype>
void GetGroundTruth(const Dtype *gt_data, const int num_gt,
                    const int background_label_id, const bool use_difficult_gt,
                    map<int, LabelBBox> *all_gt_bboxes) {
  all_gt_bboxes->clear();
  for (int i = 0; i < num_gt; ++i) {
    int start_idx = i * 8;
    int item_id = gt_data[start_idx]; // image_id
    if (item_id == -1) {
      break;
    }
    NormalizedBBox bbox;
    int label = gt_data[start_idx + 1];
    CHECK_NE(background_label_id, label)
        << "Found background label in the dataset.";
    bool difficult = static_cast<bool>(gt_data[start_idx + 7]);
    if (!use_difficult_gt && difficult) {
      // Skip reading difficult ground truth.
      continue;
    }
    bbox.set_xmin(gt_data[start_idx + 3]);
    bbox.set_ymin(gt_data[start_idx + 4]);
    bbox.set_xmax(gt_data[start_idx + 5]);
    bbox.set_ymax(gt_data[start_idx + 6]);
    bbox.set_difficult(difficult);
    float bbox_size = BBoxSize(bbox);
    bbox.set_size(bbox_size);
    (*all_gt_bboxes)[item_id][label].push_back(bbox);
  }
}

// Explicit initialization.
template void GetGroundTruth(const float *gt_data, const int num_gt,
                             const int background_label_id,
                             const bool use_difficult_gt,
                             map<int, LabelBBox> *all_gt_bboxes);
template void GetGroundTruth(const double *gt_data, const int num_gt,
                             const int background_label_id,
                             const bool use_difficult_gt,
                             map<int, LabelBBox> *all_gt_bboxes);

template <typename Dtype>
void GetLocPredictions(const Dtype *loc_data, const int num,
                       const int num_preds_per_class, const int num_loc_classes,
                       const bool share_location,
                       vector<LabelBBox> *loc_preds) {
  loc_preds->clear();
  if (share_location) {
    CHECK_EQ(num_loc_classes, 1);
  }
  loc_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    LabelBBox &label_bbox = (*loc_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_loc_classes * 4;
      for (int c = 0; c < num_loc_classes; ++c) {
        int label = share_location ? -1 : c;
        if (label_bbox.find(label) == label_bbox.end()) {
          label_bbox[label].resize(num_preds_per_class);
        }
        label_bbox[label][p].set_xmin(loc_data[start_idx + c * 4]);
        label_bbox[label][p].set_ymin(loc_data[start_idx + c * 4 + 1]);
        label_bbox[label][p].set_xmax(loc_data[start_idx + c * 4 + 2]);
        label_bbox[label][p].set_ymax(loc_data[start_idx + c * 4 + 3]);
      }
    }
    loc_data += num_preds_per_class * num_loc_classes * 4;
  }
}

// Explicit initialization.
template void GetLocPredictions(const float *loc_data, const int num,
                                const int num_preds_per_class,
                                const int num_loc_classes,
                                const bool share_location,
                                vector<LabelBBox> *loc_preds);
template void GetLocPredictions(const double *loc_data, const int num,
                                const int num_preds_per_class,
                                const int num_loc_classes,
                                const bool share_location,
                                vector<LabelBBox> *loc_preds);

template <typename Dtype>
void EncodeLocPrediction(const vector<LabelBBox> &all_loc_preds,
                         const map<int, vector<NormalizedBBox>> &all_gt_bboxes,
                         const vector<map<int, vector<int>>> &all_match_indices,
                         const vector<NormalizedBBox> &prior_bboxes,
                         const vector<vector<float>> &prior_variances,
                         const MultiBoxLossParameter &multibox_loss_param,
                         Dtype *loc_pred_data, Dtype *loc_gt_data) {
  int num = all_loc_preds.size();
  // CHECK_EQ(num, all_match_indices.size());
  // Get parameters.
  const CodeType code_type = multibox_loss_param.code_type();
  const bool encode_variance_in_target =
      multibox_loss_param.encode_variance_in_target();
  const bool bp_inside = multibox_loss_param.bp_inside();
  const bool use_prior_for_matching =
      multibox_loss_param.use_prior_for_matching();
  int count = 0;
  for (int i = 0; i < num; ++i) {
    for (auto it = all_match_indices[i].begin();
         it != all_match_indices[i].end(); ++it) {
      const int label = it->first;
      const vector<int> &match_index = it->second;
      CHECK(all_loc_preds[i].find(label) != all_loc_preds[i].end());
      const vector<NormalizedBBox> &loc_pred =
          all_loc_preds[i].find(label)->second;
      for (int j = 0; j < match_index.size(); ++j) {
        if (match_index[j] <= -1) {
          continue;
        }
        // Store encoded ground truth.
        const int gt_idx = match_index[j];
        CHECK(all_gt_bboxes.find(i) != all_gt_bboxes.end());
        CHECK_LT(gt_idx, all_gt_bboxes.find(i)->second.size());
        const NormalizedBBox &gt_bbox = all_gt_bboxes.find(i)->second[gt_idx];
        NormalizedBBox gt_encode;
        CHECK_LT(j, prior_bboxes.size());
        EncodeBBox(prior_bboxes[j], prior_variances[j], code_type,
                   encode_variance_in_target, gt_bbox, &gt_encode);
        loc_gt_data[count * 4] = gt_encode.xmin();
        loc_gt_data[count * 4 + 1] = gt_encode.ymin();
        loc_gt_data[count * 4 + 2] = gt_encode.xmax();
        loc_gt_data[count * 4 + 3] = gt_encode.ymax();
        // Store location prediction.
        CHECK_LT(j, loc_pred.size());
        if (bp_inside) {
          NormalizedBBox match_bbox = prior_bboxes[j];
          if (!use_prior_for_matching) {
            const bool clip_bbox = false;
            DecodeBBox(prior_bboxes[j], prior_variances[j], code_type,
                       encode_variance_in_target, clip_bbox, loc_pred[j],
                       &match_bbox);
          }
          // When a dimension of match_bbox is outside of image region, use
          // gt_encode to simulate zero gradient.
          loc_pred_data[count * 4] =
              (match_bbox.xmin() < 0 || match_bbox.xmin() > 1)
                  ? gt_encode.xmin()
                  : loc_pred[j].xmin();
          loc_pred_data[count * 4 + 1] =
              (match_bbox.ymin() < 0 || match_bbox.ymin() > 1)
                  ? gt_encode.ymin()
                  : loc_pred[j].ymin();
          loc_pred_data[count * 4 + 2] =
              (match_bbox.xmax() < 0 || match_bbox.xmax() > 1)
                  ? gt_encode.xmax()
                  : loc_pred[j].xmax();
          loc_pred_data[count * 4 + 3] =
              (match_bbox.ymax() < 0 || match_bbox.ymax() > 1)
                  ? gt_encode.ymax()
                  : loc_pred[j].ymax();
        } else {
          loc_pred_data[count * 4] = loc_pred[j].xmin();
          loc_pred_data[count * 4 + 1] = loc_pred[j].ymin();
          loc_pred_data[count * 4 + 2] = loc_pred[j].xmax();
          loc_pred_data[count * 4 + 3] = loc_pred[j].ymax();
        }
        if (encode_variance_in_target) {
          for (int k = 0; k < 4; ++k) {
            CHECK_GT(prior_variances[j][k], 0);
            loc_pred_data[count * 4 + k] /= prior_variances[j][k];
            loc_gt_data[count * 4 + k] /= prior_variances[j][k];
          }
        }
        ++count;
      }
    }
  }
}

// Explicit initialization.
template void
EncodeLocPrediction(const vector<LabelBBox> &all_loc_preds,
                    const map<int, vector<NormalizedBBox>> &all_gt_bboxes,
                    const vector<map<int, vector<int>>> &all_match_indices,
                    const vector<NormalizedBBox> &prior_bboxes,
                    const vector<vector<float>> &prior_variances,
                    const MultiBoxLossParameter &multibox_loss_param,
                    float *loc_pred_data, float *loc_gt_data);
template void
EncodeLocPrediction(const vector<LabelBBox> &all_loc_preds,
                    const map<int, vector<NormalizedBBox>> &all_gt_bboxes,
                    const vector<map<int, vector<int>>> &all_match_indices,
                    const vector<NormalizedBBox> &prior_bboxes,
                    const vector<vector<float>> &prior_variances,
                    const MultiBoxLossParameter &multibox_loss_param,
                    double *loc_pred_data, double *loc_gt_data);

template <typename Dtype>
void ComputeLocLoss(const Blob<Dtype> &loc_pred, const Blob<Dtype> &loc_gt,
                    const vector<map<int, vector<int>>> &all_match_indices,
                    const int num, const int num_priors,
                    const LocLossType loc_loss_type,
                    vector<vector<float>> *all_loc_loss) {
  int loc_count = loc_pred.count();
  CHECK_EQ(loc_count, loc_gt.count());
  Blob<Dtype> diff;
  const Dtype *diff_data = nullptr;
  if (loc_count != 0) {
    diff.Reshape(loc_pred.shape());
    caffe_sub(loc_count, loc_pred.cpu_data(), loc_gt.cpu_data(),
              diff.mutable_cpu_data());
    diff_data = diff.cpu_data();
  }
  int count = 0;
  for (int i = 0; i < num; ++i) {
    vector<float> loc_loss(num_priors, 0.f);
    for (const auto &it : all_match_indices[i]) {
      const vector<int> &match_index = it.second;
      CHECK_EQ(num_priors, match_index.size());
      for (int j = 0; j < match_index.size(); ++j) {
        if (match_index[j] <= -1) {
          continue;
        }
        Dtype loss = 0;
        for (int k = 0; k < 4; ++k) {
          Dtype val = diff_data[count * 4 + k];
          if (loc_loss_type == MultiBoxLossParameter_LocLossType_SMOOTH_L1) {
            Dtype abs_val = fabs(val);
            if (abs_val < 1.) {
              loss += 0.5 * val * val;
            } else {
              loss += abs_val - 0.5;
            }
          } else if (loc_loss_type == MultiBoxLossParameter_LocLossType_L2) {
            loss += 0.5 * val * val;
          } else {
            LOG(FATAL) << "Unknown loc loss type.";
          }
        }
        loc_loss[j] = loss;
        ++count;
      }
    }
    all_loc_loss->push_back(loc_loss);
  }
}

// Explicit initialization.
template void
ComputeLocLoss(const Blob<float> &loc_pred, const Blob<float> &loc_gt,
               const vector<map<int, vector<int>>> &all_match_indices,
               const int num, const int num_priors,
               const LocLossType loc_loss_type,
               vector<vector<float>> *all_loc_loss);
template void
ComputeLocLoss(const Blob<double> &loc_pred, const Blob<double> &loc_gt,
               const vector<map<int, vector<int>>> &all_match_indices,
               const int num, const int num_priors,
               const LocLossType loc_loss_type,
               vector<vector<float>> *all_loc_loss);

template <typename Dtype>
void GetConfidenceScores(const Dtype *conf_data, const int num,
                         const int num_preds_per_class, const int num_classes,
                         vector<map<int, vector<float>>> *conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    map<int, vector<float>> &label_scores = (*conf_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      for (int c = 0; c < num_classes; ++c) {
        label_scores[c].push_back(conf_data[start_idx + c]);
      }
    }
    conf_data += num_preds_per_class * num_classes;
  }
}

// Explicit initialization.
template void GetConfidenceScores(const float *conf_data, const int num,
                                  const int num_preds_per_class,
                                  const int num_classes,
                                  vector<map<int, vector<float>>> *conf_preds);
template void GetConfidenceScores(const double *conf_data, const int num,
                                  const int num_preds_per_class,
                                  const int num_classes,
                                  vector<map<int, vector<float>>> *conf_preds);

template <typename Dtype>
void GetConfidenceScores(const Dtype *conf_data, const int num,
                         const int num_preds_per_class, const int num_classes,
                         const bool class_major,
                         vector<map<int, vector<float>>> *conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    map<int, vector<float>> &label_scores = (*conf_preds)[i];
    if (class_major) {
      for (int c = 0; c < num_classes; ++c) {
        label_scores[c].assign(conf_data, conf_data + num_preds_per_class);
        conf_data += num_preds_per_class;
      }
    } else {
      for (int p = 0; p < num_preds_per_class; ++p) {
        int start_idx = p * num_classes;
        for (int c = 0; c < num_classes; ++c) {
          label_scores[c].push_back(conf_data[start_idx + c]);
        }
      }
      conf_data += num_preds_per_class * num_classes;
    }
  }
}

// Explicit initialization.
template void GetConfidenceScores(const float *conf_data, const int num,
                                  const int num_preds_per_class,
                                  const int num_classes, const bool class_major,
                                  vector<map<int, vector<float>>> *conf_preds);
template void GetConfidenceScores(const double *conf_data, const int num,
                                  const int num_preds_per_class,
                                  const int num_classes, const bool class_major,
                                  vector<map<int, vector<float>>> *conf_preds);

template <typename Dtype>
void ComputeConfLoss(const Dtype *conf_data, const int num,
                     const int num_preds_per_class, const int num_classes,
                     const int background_label_id,
                     const ConfLossType loss_type,
                     vector<vector<float>> *all_conf_loss) {
  all_conf_loss->clear();
  for (int i = 0; i < num; ++i) {
    vector<float> conf_loss;
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      int label = background_label_id;
      Dtype loss = 0;
      if (loss_type == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
        CHECK_GE(label, 0);
        CHECK_LT(label, num_classes);
        // Compute softmax probability.
        // We need to subtract the max to avoid numerical issues.
        Dtype maxval = -FLT_MAX;
        for (int c = 0; c < num_classes; ++c) {
          maxval = std::max<Dtype>(conf_data[start_idx + c], maxval);
        }
        Dtype sum = 0.;
        for (int c = 0; c < num_classes; ++c) {
          sum += std::exp(conf_data[start_idx + c] - maxval);
        }
        Dtype prob = std::exp(conf_data[start_idx + label] - maxval) / sum;
        loss = -log(std::max(prob, Dtype(FLT_MIN)));
      } else if (loss_type == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
        int target = 0;
        for (int c = 0; c < num_classes; ++c) {
          if (c == label) {
            target = 1;
          } else {
            target = 0;
          }
          Dtype input = conf_data[start_idx + c];
          loss -= input * (target - (input >= 0)) -
                  log(1 + exp(input - 2 * input * (input >= 0)));
        }
      } else {
        LOG(FATAL) << "Unknown conf loss type.";
      }
      conf_loss.push_back(loss);
    }
    conf_data += num_preds_per_class * num_classes;
    all_conf_loss->push_back(conf_loss);
  }
}

// Explicit initialization.
template void ComputeConfLoss(const float *conf_data, const int num,
                              const int num_preds_per_class,
                              const int num_classes,
                              const int background_label_id,
                              const ConfLossType loss_type,
                              vector<vector<float>> *all_conf_loss);
template void ComputeConfLoss(const double *conf_data, const int num,
                              const int num_preds_per_class,
                              const int num_classes,
                              const int background_label_id,
                              const ConfLossType loss_type,
                              vector<vector<float>> *all_conf_loss);

template <typename Dtype>
void ComputeConfLoss(const Dtype *conf_data, const int num,
                     const int num_preds_per_class, const int num_classes,
                     const int background_label_id,
                     const ConfLossType loss_type,
                     const vector<map<int, vector<int>>> &all_match_indices,
                     const map<int, vector<NormalizedBBox>> &all_gt_bboxes,
                     vector<vector<float>> *all_conf_loss) {
  CHECK_LT(background_label_id, num_classes);
  // CHECK_EQ(num, all_match_indices.size());
  all_conf_loss->clear();
  for (int i = 0; i < num; ++i) {
    vector<float> conf_loss;
    const map<int, vector<int>> &match_indices = all_match_indices[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      // Get the label index.
      int label = background_label_id;
      for (const auto &match_indice : match_indices) {
        const vector<int> &match_index = match_indice.second;
        CHECK_EQ(match_index.size(), num_preds_per_class);
        if (match_index[p] > -1) {
          CHECK(all_gt_bboxes.find(i) != all_gt_bboxes.end());
          const vector<NormalizedBBox> &gt_bboxes =
              all_gt_bboxes.find(i)->second;
          CHECK_LT(match_index[p], gt_bboxes.size());
          label = gt_bboxes[match_index[p]].label();
          CHECK_GE(label, 0);
          CHECK_NE(label, background_label_id);
          CHECK_LT(label, num_classes);
          // A prior can only be matched to one gt bbox.
          break;
        }
      }
      Dtype loss = 0;
      if (loss_type == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
        CHECK_GE(label, 0);
        CHECK_LT(label, num_classes);
        // Compute softmax probability.
        // We need to subtract the max to avoid numerical issues.
        Dtype maxval = conf_data[start_idx];
        for (int c = 1; c < num_classes; ++c) {
          maxval = std::max<Dtype>(conf_data[start_idx + c], maxval);
        }
        Dtype sum = 0.;
        for (int c = 0; c < num_classes; ++c) {
          sum += std::exp(conf_data[start_idx + c] - maxval);
        }
        Dtype prob = std::exp(conf_data[start_idx + label] - maxval) / sum;
        loss = -log(std::max(prob, Dtype(FLT_MIN)));
      } else if (loss_type == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
        int target = 0;
        for (int c = 0; c < num_classes; ++c) {
          if (c == label) {
            target = 1;
          } else {
            target = 0;
          }
          Dtype input = conf_data[start_idx + c];
          loss -= input * (target - (input >= 0)) -
                  log(1 + exp(input - 2 * input * (input >= 0)));
        }
      } else {
        LOG(FATAL) << "Unknown conf loss type.";
      }
      conf_loss.push_back(loss);
    }
    conf_data += num_preds_per_class * num_classes;
    all_conf_loss->push_back(conf_loss);
  }
}

// Explicit initialization.
template void
ComputeConfLoss(const float *conf_data, const int num,
                const int num_preds_per_class, const int num_classes,
                const int background_label_id, const ConfLossType loss_type,
                const vector<map<int, vector<int>>> &all_match_indices,
                const map<int, vector<NormalizedBBox>> &all_gt_bboxes,
                vector<vector<float>> *all_conf_loss);
template void
ComputeConfLoss(const double *conf_data, const int num,
                const int num_preds_per_class, const int num_classes,
                const int background_label_id, const ConfLossType loss_type,
                const vector<map<int, vector<int>>> &all_match_indices,
                const map<int, vector<NormalizedBBox>> &all_gt_bboxes,
                vector<vector<float>> *all_conf_loss);

template <typename Dtype>
void EncodeConfPrediction(
    const Dtype *conf_data, const int num, const int num_priors,
    const MultiBoxLossParameter &multibox_loss_param,
    const vector<map<int, vector<int>>> &all_match_indices,
    const vector<vector<int>> &all_neg_indices,
    const map<int, vector<NormalizedBBox>> &all_gt_bboxes,
    Dtype *conf_pred_data, Dtype *conf_gt_data) {
  // CHECK_EQ(num, all_match_indices.size());
  // CHECK_EQ(num, all_neg_indices.size());
  // Retrieve parameters.
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  const int num_classes = multibox_loss_param.num_classes();
  CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
  const int background_label_id = multibox_loss_param.background_label_id();
  const bool map_object_to_agnostic =
      multibox_loss_param.map_object_to_agnostic();
  if (map_object_to_agnostic) {
    if (background_label_id >= 0) {
      CHECK_EQ(num_classes, 2);
    } else {
      CHECK_EQ(num_classes, 1);
    }
  }
  const MiningType mining_type = multibox_loss_param.mining_type();
  bool do_neg_mining;
  if (multibox_loss_param.has_do_neg_mining()) {
    LOG(WARNING) << "do_neg_mining is deprecated, use mining_type instead.";
    do_neg_mining = multibox_loss_param.do_neg_mining();
    CHECK_EQ(do_neg_mining,
             mining_type != MultiBoxLossParameter_MiningType_NONE);
  }
  do_neg_mining = mining_type != MultiBoxLossParameter_MiningType_NONE;
  const ConfLossType conf_loss_type = multibox_loss_param.conf_loss_type();
  int count = 0;
  for (int i = 0; i < num; ++i) {
    if (all_gt_bboxes.find(i) != all_gt_bboxes.end()) {
      // Save matched (positive) bboxes scores and labels.
      const map<int, vector<int>> &match_indices = all_match_indices[i];
      for (const auto &match_indice : match_indices) {
        const vector<int> &match_index = match_indice.second;
        CHECK_EQ(match_index.size(), num_priors);
        for (int j = 0; j < num_priors; ++j) {
          if (match_index[j] <= -1) {
            continue;
          }
          const int gt_label =
              map_object_to_agnostic
                  ? background_label_id + 1
                  : all_gt_bboxes.find(i)->second[match_index[j]].label();
          int idx = do_neg_mining ? count : j;
          switch (conf_loss_type) {
          case MultiBoxLossParameter_ConfLossType_SOFTMAX:
            conf_gt_data[idx] = gt_label;
            break;
          case MultiBoxLossParameter_ConfLossType_LOGISTIC:
            conf_gt_data[idx * num_classes + gt_label] = 1;
            break;
          default:
            LOG(FATAL) << "Unknown conf loss type.";
          }
          if (do_neg_mining) {
            // Copy scores for matched bboxes.
            caffe_copy<Dtype>(num_classes, conf_data + j * num_classes,
                              conf_pred_data + count * num_classes);
            ++count;
          }
        }
      }
      // Go to next image.
      if (do_neg_mining) {
        // Save negative bboxes scores and labels.
        for (int j : all_neg_indices[i]) {
          CHECK_LT(j, num_priors);
          caffe_copy<Dtype>(num_classes, conf_data + j * num_classes,
                            conf_pred_data + count * num_classes);
          switch (conf_loss_type) {
          case MultiBoxLossParameter_ConfLossType_SOFTMAX:
            conf_gt_data[count] = background_label_id;
            break;
          case MultiBoxLossParameter_ConfLossType_LOGISTIC:
            if (background_label_id >= 0 && background_label_id < num_classes) {
              conf_gt_data[count * num_classes + background_label_id] = 1;
            }
            break;
          default:
            LOG(FATAL) << "Unknown conf loss type.";
          }
          ++count;
        }
      }
    }
    if (do_neg_mining) {
      conf_data += num_priors * num_classes;
    } else {
      conf_gt_data += num_priors;
    }
  }
}

// Explicite initialization.
template void
EncodeConfPrediction(const float *conf_data, const int num,
                     const int num_priors,
                     const MultiBoxLossParameter &multibox_loss_param,
                     const vector<map<int, vector<int>>> &all_match_indices,
                     const vector<vector<int>> &all_neg_indices,
                     const map<int, vector<NormalizedBBox>> &all_gt_bboxes,
                     float *conf_pred_data, float *conf_gt_data);
template void
EncodeConfPrediction(const double *conf_data, const int num,
                     const int num_priors,
                     const MultiBoxLossParameter &multibox_loss_param,
                     const vector<map<int, vector<int>>> &all_match_indices,
                     const vector<vector<int>> &all_neg_indices,
                     const map<int, vector<NormalizedBBox>> &all_gt_bboxes,
                     double *conf_pred_data, double *conf_gt_data);

template <typename Dtype>
void GetPriorBBoxes(const Dtype *prior_data, const int num_priors,
                    vector<NormalizedBBox> *prior_bboxes,
                    vector<vector<float>> *prior_variances) {
  prior_bboxes->clear();
  prior_variances->clear();
  for (int i = 0; i < num_priors; ++i) {
    int start_idx = i * 4;
    NormalizedBBox bbox;
    bbox.set_xmin(prior_data[start_idx]);
    bbox.set_ymin(prior_data[start_idx + 1]);
    bbox.set_xmax(prior_data[start_idx + 2]);
    bbox.set_ymax(prior_data[start_idx + 3]);
    float bbox_size = BBoxSize(bbox);
    bbox.set_size(bbox_size);
    prior_bboxes->push_back(bbox);
  }

  for (int i = 0; i < num_priors; ++i) {
    int start_idx = (num_priors + i) * 4;
    vector<float> var;
    for (int j = 0; j < 4; ++j) {
      var.push_back(prior_data[start_idx + j]);
    }
    prior_variances->push_back(var);
  }
}

// Explicit initialization.
template void GetPriorBBoxes(const float *prior_data, const int num_priors,
                             vector<NormalizedBBox> *prior_bboxes,
                             vector<vector<float>> *prior_variances);
template void GetPriorBBoxes(const double *prior_data, const int num_priors,
                             vector<NormalizedBBox> *prior_bboxes,
                             vector<vector<float>> *prior_variances);

template <typename Dtype>
void GetDetectionResults(
    const Dtype *det_data, const int num_det, const int background_label_id,
    map<int, map<int, vector<NormalizedBBox>>> *all_detections) {
  all_detections->clear();
  for (int i = 0; i < num_det; ++i) {
    int start_idx = i * 7;
    int item_id = det_data[start_idx]; // image_id
    if (item_id == -1) {
      continue;
    }
    int label = det_data[start_idx + 1];
    CHECK_NE(background_label_id, label)
        << "Found background label in the detection results.";
    NormalizedBBox bbox;
    bbox.set_score(det_data[start_idx + 2]);
    bbox.set_xmin(det_data[start_idx + 3]);
    bbox.set_ymin(det_data[start_idx + 4]);
    bbox.set_xmax(det_data[start_idx + 5]);
    bbox.set_ymax(det_data[start_idx + 6]);
    float bbox_size = BBoxSize(bbox);
    bbox.set_size(bbox_size);
    (*all_detections)[item_id][label].push_back(bbox);
  }
}

// Explicit initialization.
template void
GetDetectionResults(const float *det_data, const int num_det,
                    const int background_label_id,
                    map<int, map<int, vector<NormalizedBBox>>> *all_detections);
template void
GetDetectionResults(const double *det_data, const int num_det,
                    const int background_label_id,
                    map<int, map<int, vector<NormalizedBBox>>> *all_detections);

void GetTopKScoreIndex(const vector<float> &scores, const vector<int> &indices,
                       const int top_k,
                       vector<pair<float, int>> *score_index_vec) {
  CHECK_EQ(scores.size(), indices.size());

  // Generate index score pairs.
  for (int i = 0; i < scores.size(); ++i) {
    score_index_vec->push_back(std::make_pair(scores[i], indices[i]));
  }

  // Sort the score pair according to the scores in descending order
  std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                   SortScorePairDescend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}

void GetMaxScoreIndex(const vector<float> &scores, const float threshold,
                      const int top_k,
                      vector<pair<float, int>> *score_index_vec) {
  // Generate index score pairs.
  for (int i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      score_index_vec->push_back(std::make_pair(scores[i], i));
    }
  }

  // Sort the score pair according to the scores in descending order
  std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                   SortScorePairDescend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}

template <typename Dtype>
void GetMaxScoreIndex(const Dtype *scores, const int num, const float threshold,
                      const int top_k,
                      vector<pair<Dtype, int>> *score_index_vec) {
  // Generate index score pairs.
  for (int i = 0; i < num; ++i) {
    if (scores[i] > threshold) {
      score_index_vec->push_back(std::make_pair(scores[i], i));
    }
  }

  // Sort the score pair according to the scores in descending order
  std::sort(score_index_vec->begin(), score_index_vec->end(),
            SortScorePairDescend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}

template void GetMaxScoreIndex(const float *scores, const int num,
                               const float threshold, const int top_k,
                               vector<pair<float, int>> *score_index_vec);
template void GetMaxScoreIndex(const double *scores, const int num,
                               const float threshold, const int top_k,
                               vector<pair<double, int>> *score_index_vec);

void ApplyNMS(const vector<NormalizedBBox> &bboxes, const vector<float> &scores,
              const float threshold, const int top_k, const bool reuse_overlaps,
              map<int, map<int, float>> *overlaps, vector<int> *indices) {
  // Sanity check.
  CHECK_EQ(bboxes.size(), scores.size())
      << "bboxes and scores have different size.";

  // Get top_k scores (with corresponding indices).
  vector<int> idx(boost::counting_iterator<int>(0),
                  boost::counting_iterator<int>(scores.size()));
  vector<pair<float, int>> score_index_vec;
  GetTopKScoreIndex(scores, idx, top_k, &score_index_vec);

  // Do nms.
  indices->clear();
  while (!score_index_vec.empty()) {
    // Get the current highest score box.
    int best_idx = score_index_vec.front().second;
    const NormalizedBBox &best_bbox = bboxes[best_idx];
    if (BBoxSize(best_bbox) < 1e-5) {
      // Erase small box.
      score_index_vec.erase(score_index_vec.begin());
      continue;
    }
    indices->push_back(best_idx);
    // Erase the best box.
    score_index_vec.erase(score_index_vec.begin());

    if (top_k > -1 && indices->size() >= top_k) {
      // Stop if finding enough bboxes for nms.
      break;
    }

    // Compute overlap between best_bbox and other remaining bboxes.
    // Remove a bbox if the overlap with best_bbox is larger than nms_threshold.
    for (auto it = score_index_vec.begin(); it != score_index_vec.end();) {
      int cur_idx = it->second;
      const NormalizedBBox &cur_bbox = bboxes[cur_idx];
      if (BBoxSize(cur_bbox) < 1e-5) {
        // Erase small box.
        it = score_index_vec.erase(it);
        continue;
      }
      float cur_overlap = 0.;
      if (reuse_overlaps) {
        if (overlaps->find(best_idx) != overlaps->end() &&
            overlaps->find(best_idx)->second.find(cur_idx) !=
                (*overlaps)[best_idx].end()) {
          // Use the computed overlap.
          cur_overlap = (*overlaps)[best_idx][cur_idx];
        } else if (overlaps->find(cur_idx) != overlaps->end() &&
                   overlaps->find(cur_idx)->second.find(best_idx) !=
                       (*overlaps)[cur_idx].end()) {
          // Use the computed overlap.
          cur_overlap = (*overlaps)[cur_idx][best_idx];
        } else {
          cur_overlap = JaccardOverlap(best_bbox, cur_bbox);
          // Store the overlap for future use.
          (*overlaps)[best_idx][cur_idx] = cur_overlap;
        }
      } else {
        cur_overlap = JaccardOverlap(best_bbox, cur_bbox);
      }

      // Remove it if necessary
      if (cur_overlap > threshold) {
        it = score_index_vec.erase(it);
      } else {
        ++it;
      }
    }
  }
}

void ApplyNMS(const vector<NormalizedBBox> &bboxes, const vector<float> &scores,
              const float threshold, const int top_k, vector<int> *indices) {
  bool reuse_overlap = false;
  map<int, map<int, float>> overlaps;
  ApplyNMS(bboxes, scores, threshold, top_k, reuse_overlap, &overlaps, indices);
}

void ApplyNMS(const bool *overlapped, const int num, vector<int> *indices) {
  vector<int> index_vec(boost::counting_iterator<int>(0),
                        boost::counting_iterator<int>(num));
  // Do nms.
  indices->clear();
  while (!index_vec.empty()) {
    // Get the current highest score box.
    int best_idx = index_vec.front();
    indices->push_back(best_idx);
    // Erase the best box.
    index_vec.erase(index_vec.begin());

    for (auto it = index_vec.begin(); it != index_vec.end();) {
      int cur_idx = *it;

      // Remove it if necessary
      if (overlapped[best_idx * num + cur_idx]) {
        it = index_vec.erase(it);
      } else {
        ++it;
      }
    }
  }
}

inline int clamp(const int v, const int a, const int b) {
  return v < a ? a : v > b ? b : v;
}

void ApplyNMSFast(const vector<NormalizedBBox> &bboxes,
                  const vector<float> &scores, const float score_threshold,
                  const float nms_threshold, const float eta, const int top_k,
                  vector<int> *indices) {
  // Sanity check.
  CHECK_EQ(bboxes.size(), scores.size())
      << "bboxes and scores have different size.";

  // Get top_k scores (with corresponding indices).
  vector<pair<float, int>> score_index_vec;
  GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);

  // Do nms.
  float adaptive_threshold = nms_threshold;
  indices->clear();
  while (!score_index_vec.empty()) {
    const int idx = score_index_vec.front().second;
    bool keep = true;
    for (int kept_idx : *indices) {
      if (keep) {
        float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
}

template <typename Dtype>
void ApplyNMSFast(const Dtype *bboxes, const Dtype *scores, const int num,
                  const float score_threshold, const float nms_threshold,
                  const float eta, const int top_k, vector<int> *indices) {
  // Get top_k scores (with corresponding indices).
  vector<pair<Dtype, int>> score_index_vec;
  GetMaxScoreIndex(scores, num, score_threshold, top_k, &score_index_vec);

  // Do nms.
  float adaptive_threshold = nms_threshold;
  indices->clear();
  while (!score_index_vec.empty()) {
    const int idx = score_index_vec.front().second;
    bool keep = true;
    for (int kept_idx : *indices) {
      if (keep) {
        float overlap = JaccardOverlap(bboxes + idx * 4, bboxes + kept_idx * 4);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
}

template void ApplyNMSFast(const float *bboxes, const float *scores,
                           const int num, const float score_threshold,
                           const float nms_threshold, const float eta,
                           const int top_k, vector<int> *indices);
template void ApplyNMSFast(const double *bboxes, const double *scores,
                           const int num, const float score_threshold,
                           const float nms_threshold, const float eta,
                           const int top_k, vector<int> *indices);

void CumSum(const vector<pair<float, int>> &pairs, vector<int> *cumsum) {
  // Sort the pairs based on first item of the pair.
  vector<pair<float, int>> sort_pairs = pairs;
  std::stable_sort(sort_pairs.begin(), sort_pairs.end(),
                   SortScorePairDescend<int>);

  cumsum->clear();
  for (int i = 0; i < sort_pairs.size(); ++i) {
    if (i == 0) {
      cumsum->push_back(sort_pairs[i].second);
    } else {
      cumsum->push_back(cumsum->back() + sort_pairs[i].second);
    }
  }
}

void ComputeAP(const vector<pair<float, int>> &tp, const int num_pos,
               const vector<pair<float, int>> &fp, const string &ap_version,
               vector<float> *prec, vector<float> *rec, float *ap) {
  const float eps = 1e-6;
  CHECK_EQ(tp.size(), fp.size()) << "tp must have same size as fp.";
  const int num = tp.size();
  // Make sure that tp and fp have complement value.
  for (int i = 0; i < num; ++i) {
    CHECK_LE(fabs(tp[i].first - fp[i].first), eps);
    CHECK_EQ(tp[i].second, 1 - fp[i].second);
  }
  prec->clear();
  rec->clear();
  *ap = 0;
  if (tp.empty() || num_pos == 0) {
    return;
  }

  // Compute cumsum of tp.
  vector<int> tp_cumsum;
  CumSum(tp, &tp_cumsum);
  CHECK_EQ(tp_cumsum.size(), num);

  // Compute cumsum of fp.
  vector<int> fp_cumsum;
  CumSum(fp, &fp_cumsum);
  CHECK_EQ(fp_cumsum.size(), num);

  // Compute precision.
  for (int i = 0; i < num; ++i) {
    prec->push_back(static_cast<float>(tp_cumsum[i]) /
                    (tp_cumsum[i] + fp_cumsum[i]));
  }

  // Compute recall.
  for (int i = 0; i < num; ++i) {
    CHECK_LE(tp_cumsum[i], num_pos);
    rec->push_back(static_cast<float>(tp_cumsum[i]) / num_pos);
  }

  if (ap_version == "11point") {
    // VOC2007 style for computing AP.
    vector<float> max_precs(11, 0.);
    int start_idx = num - 1;
    for (int j = 10; j >= 0; --j) {
      for (int i = start_idx; i >= 0; --i) {
        if ((*rec)[i] < j / 10.) {
          start_idx = i;
          if (j > 0) {
            max_precs[j - 1] = max_precs[j];
          }
          break;
        } else {
          if (max_precs[j] < (*prec)[i]) {
            max_precs[j] = (*prec)[i];
          }
        }
      }
    }
    for (int j = 10; j >= 0; --j) {
      *ap += max_precs[j] / 11;
    }
  } else if (ap_version == "MaxIntegral") {
    // VOC2012 or ILSVRC style for computing AP.
    float cur_rec = rec->back();
    float cur_prec = prec->back();
    for (int i = num - 2; i >= 0; --i) {
      cur_prec = std::max<float>((*prec)[i], cur_prec);
      if (fabs(cur_rec - (*rec)[i]) > eps) {
        *ap += cur_prec * fabs(cur_rec - (*rec)[i]);
      }
      cur_rec = (*rec)[i];
    }
    *ap += cur_rec * cur_prec;
  } else if (ap_version == "Integral") {
    // Natural integral.
    float prev_rec = 0.;
    for (int i = 0; i < num; ++i) {
      if (fabs((*rec)[i] - prev_rec) > eps) {
        *ap += (*prec)[i] * fabs((*rec)[i] - prev_rec);
      }
      prev_rec = (*rec)[i];
    }
  } else {
    LOG(FATAL) << "Unknown ap_version: " << ap_version;
  }
}

#ifdef USE_OPENCV
cv::Scalar HSV2RGB(const float h, const float s, const float v) {
  const int h_i = static_cast<int>(h * 6);
  const float f = h * 6 - h_i;
  const float p = v * (1 - s);
  const float q = v * (1 - f * s);
  const float t = v * (1 - (1 - f) * s);
  float r, g, b;
  switch (h_i) {
  case 0:
    r = v;
    g = t;
    b = p;
    break;
  case 1:
    r = q;
    g = v;
    b = p;
    break;
  case 2:
    r = p;
    g = v;
    b = t;
    break;
  case 3:
    r = p;
    g = q;
    b = v;
    break;
  case 4:
    r = t;
    g = p;
    b = v;
    break;
  case 5:
    r = v;
    g = p;
    b = q;
    break;
  default:
    r = 1;
    g = 1;
    b = 1;
    break;
  }
  return cv::Scalar(r * 255, g * 255, b * 255);
}

// http://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically
vector<cv::Scalar> GetColors(const int n) {
  vector<cv::Scalar> colors;
  cv::RNG rng(12345);
  const float golden_ratio_conjugate = 0.618033988749895;
  const float s = 0.3;
  const float v = 0.99;
  for (int i = 0; i < n; ++i) {
    const float h =
        std::fmod(rng.uniform(0.f, 1.f) + golden_ratio_conjugate, 1.f);
    colors.push_back(HSV2RGB(h, s, v));
  }
  return colors;
}

static clock_t start_clock = clock();
static cv::VideoWriter cap_out;

template <typename Dtype>
void VisualizeBBox(const vector<cv::Mat> &images, const Blob<Dtype> *detections,
                   const float threshold, const vector<cv::Scalar> &colors,
                   const map<int, string> &label_to_display_name,
                   const string &save_file) {
  // Retrieve detections.
  CHECK_EQ(detections->width(), 7);
  const int num_det = detections->height();
  const int num_img = images.size();
  if (num_det == 0 || num_img == 0) {
    return;
  }
  // Comute FPS.
  float fps =
      num_img / (static_cast<double>(clock() - start_clock) / CLOCKS_PER_SEC);

  const Dtype *detections_data = detections->cpu_data();
  const int width = images[0].cols;
  const int height = images[0].rows;
  vector<LabelBBox> all_detections(num_img);
  for (int i = 0; i < num_det; ++i) {
    const int img_idx = detections_data[i * 7];
    CHECK_LT(img_idx, num_img);
    const int label = detections_data[i * 7 + 1];
    const float score = detections_data[i * 7 + 2];
    if (score < threshold) {
      continue;
    }
    NormalizedBBox bbox;
    bbox.set_xmin(detections_data[i * 7 + 3] * width);
    bbox.set_ymin(detections_data[i * 7 + 4] * height);
    bbox.set_xmax(detections_data[i * 7 + 5] * width);
    bbox.set_ymax(detections_data[i * 7 + 6] * height);
    bbox.set_score(score);
    all_detections[img_idx][label].push_back(bbox);
  }

  int fontface = cv::FONT_HERSHEY_SIMPLEX;
  double scale = 1;
  int thickness = 2;
  int baseline = 0;
  char buffer[50];
  for (int i = 0; i < num_img; ++i) {
    cv::Mat image = images[i];
    // Show FPS.
    snprintf(buffer, sizeof(buffer), "FPS: %.2f", fps);
    cv::Size text =
        cv::getTextSize(buffer, fontface, scale, thickness, &baseline);
    int cvFilled = 0;
#if (!defined(CV_VERSION_EPOCH) && CV_VERSION_MAJOR >= 3)
    cvFilled = cv::FILLED;
#else
    cvFilled = CV_FILLED;
#endif

    cv::rectangle(image, cv::Point(0, 0),
                  cv::Point(text.width, text.height + baseline),
                  CV_RGB(255, 255, 255), cvFilled);
    cv::putText(image, buffer, cv::Point(0, text.height + baseline / 2.),
                fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
    // Draw bboxes.
    for (auto &it : all_detections[i]) {
      int label = it.first;
      string label_name = "Unknown";
      if (label_to_display_name.find(label) != label_to_display_name.end()) {
        label_name = label_to_display_name.find(label)->second;
      }
      CHECK_LT(label, colors.size());
      const cv::Scalar &color = colors[label];
      const vector<NormalizedBBox> &bboxes = it.second;
      for (const auto &bboxe : bboxes) {
        cv::Point top_left_pt(bboxe.xmin(), bboxe.ymin());
        cv::Point bottom_right_pt(bboxe.xmax(), bboxe.ymax());
        cv::rectangle(image, top_left_pt, bottom_right_pt, color, 4);
        cv::Point bottom_left_pt(bboxe.xmin(), bboxe.ymax());
        snprintf(buffer, sizeof(buffer), "%s: %.2f", label_name.c_str(),
                 bboxe.score());
        cv::Size text =
            cv::getTextSize(buffer, fontface, scale, thickness, &baseline);
        cv::rectangle(image, bottom_left_pt + cv::Point(0, 0),
                      bottom_left_pt +
                          cv::Point(text.width, -text.height - baseline),
                      color, cvFilled);
        cv::putText(image, buffer, bottom_left_pt - cv::Point(0, baseline),
                    fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
      }
    }
    // Save result if required.
    if (!save_file.empty()) {
      if (!cap_out.isOpened()) {
        cv::Size size(image.size().width, image.size().height);
#if (!defined(CV_VERSION_EPOCH) && CV_VERSION_MAJOR >= 3)
        cv::VideoWriter outputVideo(save_file,
                                    cv::VideoWriter::fourcc('D', 'I', 'V', 'X'),
                                    30, size, true);
#else
        cv::VideoWriter outputVideo(save_file, CV_FOURCC('D', 'I', 'V', 'X'),
                                    30, size, true);
#endif
        cap_out = outputVideo;
      }
      cap_out.write(image);
    }
    cv::imshow("detections", image);
    if (cv::waitKey(1) == 27) {
      raise(SIGINT);
    }
  }
  start_clock = clock();
}

template void VisualizeBBox(const vector<cv::Mat> &images,
                            const Blob<float> *detections,
                            const float threshold,
                            const vector<cv::Scalar> &colors,
                            const map<int, string> &label_to_display_name,
                            const string &save_file);
template void VisualizeBBox(const vector<cv::Mat> &images,
                            const Blob<double> *detections,
                            const float threshold,
                            const vector<cv::Scalar> &colors,
                            const map<int, string> &label_to_display_name,
                            const string &save_file);

using google::protobuf::RepeatedPtrField;
void MakeMosaic(
    const vector<pair<cv::Mat, RepeatedPtrField<AnnotationGroup>>> &raw, int s,
    int channels, AnnotatedDatum *anno_datum) {
  anno_datum->Clear();
  cv::Mat out(s * 2, s * 2, CV_8UC(channels),
              channels == 3 ? cv::Scalar(128, 128, 128) : cv::Scalar(128));
  float center[2], ratio;
  caffe_rng_uniform<float>(2, static_cast<float>(s) * 0.5,
                           static_cast<float>(s) * 1.5, center);
  int xc = static_cast<int>(center[0]);
  int yc = static_cast<int>(center[1]);
  auto *out_groups = anno_datum->mutable_annotation_group();
  for (int i = 0; i < 4; ++i) {
    auto [img, anno_groups] = raw[i];
    int h = img.rows;
    int w = img.cols;
    ratio = static_cast<float>(s) / std::max(h, w);
    cv::resize(
        img, img,
        cv::Size(static_cast<int>(w * ratio), static_cast<int>(h * ratio)));
    h = img.rows;
    w = img.cols;

    int x1a, y1a, x2a, y2a; // big
    int x1b, y1b, x2b, y2b; // small
    int padw, padh;
    if (i == 0) {
      x1a = std::max(xc - w, 0);
      y1a = std::max(yc - h, 0);
      x2a = xc;
      y2a = yc;
      x1b = w - (x2a - x1a);
      y1b = h - (y2a - y1a);
      x2b = w;
      y2b = h;
      padw = xc - w;
      padh = yc - h;
    } else if (i == 1) {
      x1a = xc;
      y1a = std::max(yc - h, 0);
      x2a = std::min(xc + w, s * 2);
      y2a = yc;
      x1b = 0;
      y1b = h - (y2a - y1a);
      x2b = std::min(w, x2a - x1a);
      y2b = h;
      padw = xc;
      padh = yc - h;
    } else if (i == 2) {
      x1a = std::max(xc - w, 0);
      y1a = yc;
      x2a = xc;
      y2a = std::min(s * 2, yc + h);
      x1b = w - (x2a - x1a);
      y1b = 0;
      // x2b = std::max(xc, w);
      x2b = w;
      y2b = std::min(y2a - y1a, h);
      padw = xc - w;
      padh = yc;
    } else if (i == 3) {
      x1a = xc;
      y1a = yc;
      x2a = std::min(xc + w, s * 2);
      y2a = std::min(yc + h, s * 2);
      x1b = 0;
      y1b = 0;
      x2b = std::min(x2a - x1a, w);
      y2b = std::min(y2a - y1a, h);
      padw = xc;
      padh = yc;
    }
    cv::Rect img_roi(x1b, y1b, x2b - x1b, y2b - y1b);
    cv::Rect out_roi(x1a, y1a, x2a - x1a, y2a - y1a);
    CHECK_EQ(img_roi.width, out_roi.width);
    CHECK_EQ(img_roi.height, out_roi.height);
    img(img_roi).copyTo(out(out_roi));
    cv::imwrite("/Users/troy/out.jpg", out);
    for (const auto &group : anno_groups) {
      AnnotationGroup filtered_group;
      filtered_group.set_group_label(group.group_label());
      for (auto anno : group.annotation()) {
        auto *bbox = anno.mutable_bbox();
        float xmin = bbox->xmin() * w + padw;
        float ymin = bbox->ymin() * h + padh;
        float xmax = bbox->xmax() * w + padw;
        float ymax = bbox->ymax() * h + padh;
        bbox->set_xmin(caffe_clip<float>(xmin, 0, 2 * s) / (2 * s));
        bbox->set_ymin(caffe_clip<float>(ymin, 0, 2 * s) / (2 * s));
        bbox->set_xmax(caffe_clip<float>(xmax, 0, 2 * s) / (2 * s));
        bbox->set_ymax(caffe_clip<float>(ymax, 0, 2 * s) / (2 * s));
        if (BBoxSize(*bbox) > 0) {
          *(filtered_group.add_annotation()) = anno;
        }
      }
      if (filtered_group.annotation_size() > 0) {
        out_groups->Add(std::move(filtered_group));
      }
    }
  }
  CVMatToDatum(out, anno_datum->mutable_datum());
}

#endif // USE_OPENCV

} // namespace caffe
