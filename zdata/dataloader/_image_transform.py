import cv2
import numpy as np

# cv2.setNumThreads(1)
cv2.setNumThreads(2)


def _affine_to_perspective(t):
    return np.concatenate([t, [[0, 0, 1]]], axis=0)


def _point2d_to_homo(p):
    s = p.shape
    assert s[-1] == 2, "the last dim must be 2"
    p = p.reshape([-1, 2])
    q = np.concatenate([p, np.ones([p.shape[0], 1], dtype=p.dtype)], axis=1)
    q = np.reshape(q, list(s[:-1]) + [3])
    return q


def _homo_to_point2d(p):
    s = p.shape
    assert s[-1] == 3, "the last dim must be 3"
    p = p.reshape([-1, 3])
    q = p[:, :2] / p[:, 2:3]
    q = np.reshape(q, list(s[:-1]) + [2])
    return q


def unnormalized_transform_to_normalized_transform(t, dst_w, dst_h, src_w, src_h):
    t = np.array(t)
    assert len(t.shape) == 2 and t.shape[1] == 3 and t.shape[0] in (2, 3), "t must be a {2,3}x3 matrix"
    is_affine = (t.shape[0] == 2)
    if is_affine:
        t = _affine_to_perspective(t)

    a = np.array([      # src normalized <- src unnormalized
        [1/src_w, 0, 0],
        [0, 1/src_h, 0],
        [0, 0, 1]
    ], dtype=t.dtype)
    # t: src unnormalized <- dst unnormalized
    b = np.array([          # dst unnormalized <- dst normalized
        [dst_w, 0, 0],
        [0, dst_h, 0],
        [0, 0, 1]
    ], dtype=t.dtype)

    p = np.dot(a, np.dot(t, b))
    if is_affine:
        p = p[:2]
    return p


def normalized_transform_to_unnormalized_transform(t, dst_w, dst_h, src_w, src_h):
    t = np.array(t)
    assert len(t.shape) == 2 and t.shape[1] == 3 and t.shape[0] in (2, 3), "t must be a {2,3}x3 matrix"
    is_affine = (t.shape[0] == 2)
    if is_affine:
        t = _affine_to_perspective(t)

    a = np.array([      # src unnormalized <- src normalized
        [src_w, 0, 0],
        [0, src_h, 0],
        [0, 0, 1]
    ], dtype=t.dtype)
    # t: src normalized <- dst normalized
    b = np.array([          # dst normalized <- dst unnormalized
        [1/dst_w, 0, 0],
        [0, 1/dst_h, 0],
        [0, 0, 1]
    ], dtype=t.dtype)

    p = np.dot(a, np.dot(t, b))
    if is_affine:
        p = p[:2]
    return p


def warp_affine(im, t, output_wh):
    t = np.array(t)
    im = np.array(im)
    if t[0, 1] == 0 and t[1, 0] == 0:
        # no rotation
        out_w, out_h = output_wh
        if t[0, 2] == 0 and t[1, 2] == 0:
            # no offset
            a = cv2.resize(im, None, fx=1/t[0, 0], fy=1/t[1,1])
            padded_im_shape = \
                [max(out_h, a.shape[0]), max(out_w, a.shape[1])] + \
                list(a.shape[2:]) if len(a.shape) > 2 else []
            b = np.zeros(padded_im_shape, dtype=a.dtype)
            b[:a.shape[0], :a.shape[1]] = a
            out_im = b[:out_h, :out_w]
        else:
            if True:
                # out_im = cv2.warpAffine(im, t, output_wh, flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                out_im = cv2.warpAffine(im, t, output_wh, flags=cv2.WARP_INVERSE_MAP)
            else:
                a = cv2.resize(im, None, fx=1 / t[0, 0], fy=1 / t[1, 1])
                t_x = t[0, 2] / t[0, 0]
                t_y = t[1, 2] / t[1, 1]
                out_t = np.eye(2, 3, dtype=t.dtype)
                out_t[:, 2] = (t_x, t_y)
                out_im = cv2.warpAffine(
                    a, out_t, output_wh, flags=cv2.WARP_INVERSE_MAP,
                )
    else:
        # out_im = cv2.warpAffine(im, t, output_wh, flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        out_im = cv2.warpAffine(im, t, output_wh, flags=cv2.WARP_INVERSE_MAP)
    return np.array(out_im).astype(im.dtype)


def affine_im(im, t, output_wh):
    im = np.array(im)
    out_im = warp_affine(im, t, output_wh)
    return np.array(out_im).astype(im.dtype)


def affine_flow(f, t, t_next, output_wh, dim_order="xy", is_normalized=False, im_wh=None):
    f0 = np.array(f)
    f = f0
    assert len(f.shape) == 3 and f.shape[-1] == 2, "the flow shape must be (H, W, 2)"
    if dim_order == "xy":
        pass
    elif dim_order == "yx":
        f = np.flip(f, axis=-1)
    else:
        raise ValueError("Invalid dim_order")

    if is_normalized:
        im_w, im_h = im_wh
        f *= np.array([im_w, im_h], dtype=f.dtype).reshape([1, 1, 2])

    # decomposition transform
    t = np.array(t, dtype=f.dtype)
    t_next = np.array(t_next, dtype=f.dtype)
    r1 = t[:, :2]
    b1 = np.expand_dims(t[:, 2], axis=-1)
    r2 = t_next[:, :2]
    b2 = np.expand_dims(t_next[:, 2], axis=-1)

    w0, h0 = output_wh

    # crop flow
    h = warp_affine(f, t, (w0, h0))
    h = np.reshape(h, [-1, 2]).T

    # scale and rotate flow
    h = np.linalg.lstsq(r2, h, rcond=None)[0]

    # compute flow offset
    c = np.linalg.lstsq(r2, r1, rcond=None)[0] - np.eye(2, 2, dtype=f.dtype)
    xg, yg = np.meshgrid(
        np.array(range(w0), dtype=f.dtype),
        np.array(range(h0), dtype=f.dtype),
        indexing='xy'
    )
    g = np.concatenate(
        [xg[np.newaxis, :, :], yg[np.newaxis, :, :]], axis=0
    )
    d = np.dot(c, np.reshape(g, [2, -1]))
    e = np.linalg.lstsq(r2, b1 - b2, rcond=None)[0]

    h += d + e

    if is_normalized:
        h *= np.array([1 / w0, 1 / h0], dtype=h.dtype)[np.newaxis, :]

    h = h.T

    if dim_order == "yx":
        h = np.flip(h, axis=-1)

    h = np.reshape(h, [h0, w0, 2])

    return h.astype(f0.dtype)


def affine_point(p, t, output_wh, dim_order="xy", is_normalized=False, im_wh=None, ):
    p0 = np.array(p)
    assert p0.shape[-1] == 2, "the last dim must be two"
    p = p0.reshape([-1, 2])
    if dim_order == "xy":
        pass
    elif dim_order == "yx":
        p = np.flip(p, axis=-1)
    else:
        raise ValueError("Invalid dim_order")

    if is_normalized:
        im_w, im_h = im_wh
        p *= np.array([im_w, im_h], dtype=p.dtype).reshape([1, 2])

    q = np.linalg.lstsq(_affine_to_perspective(t), _point2d_to_homo(p).T, rcond=None)[0].T
    q = _homo_to_point2d(q)

    if is_normalized:
        w0, h0 = output_wh
        q *= np.array([1 / w0, 1 / h0], dtype=p.dtype).reshape([1, 2])

    if dim_order == "yx":
        q = np.flip(q, axis=-1)

    q = q.reshape(p0.shape)

    return q.astype(p0.dtype)


def affine_box(b, t, output_wh, dim_order="xywh", is_normalized=False, im_wh=None):
    b0 = np.array(b)
    assert b0.shape[-1] == 4, "the last dim must be four"
    b = b0.reshape([-1, 4])
    if dim_order in ("xywh", "yxhw"):
        b = np.concatenate([
            b[:, :2], b[:, 0:2] + b[:, 2:4]
        ], axis=-1)
    elif dim_order in ("xyxy", "yxyx"):
        pass
    else:
        raise ValueError("Invalid dim_order")
    b = b.reshape([-1, 2, 2])
    b = np.concatenate([
        b,
        np.concatenate([b[:, 0:1, 0:1], b[:, 1:2, 1:2]], axis=2),
        np.concatenate([b[:, 1:2, 0:1], b[:, 0:1, 1:2]], axis=2),
    ], axis=1)  # extend to 4-pts, shape: [-1, 4, 2], box will be enlarged after rotation to include the whole region
    c = affine_point(
        b, t, output_wh=output_wh,
        dim_order=dim_order[:2], is_normalized=is_normalized, im_wh=im_wh
    )  # shape: [-1, 4, 2]
    c = np.concatenate([np.min(c, axis=1), np.max(c, axis=1)], axis=1) # shape: [-1, 4]
    if dim_order in ("xywh", "yxhw"):
        c = np.concatenate([
            c[:, 0:2], c[:, 2:4] - c[:, 0:2]
        ], axis=-1)
    c = c.reshape(b0.shape)
    return c.astype(b0.dtype)


def flow_to_rgb(flow):
    flow = np.array(flow)
    hsv = np.zeros([flow.shape[0], flow.shape[1], 3], dtype=flow.dtype)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb
