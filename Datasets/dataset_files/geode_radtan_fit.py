"""Approximate GEODE Brown radtan5 by radtan4 in pixel projection space.

Fits known calibration mappings, never image features, trajectories or GT.
The selected fit adjusts fx, fy, cx, cy and all four distortion coefficients.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares


def source_parameters(camera):
    p, d = camera['projection_parameters'], camera['distortion_parameters']
    K = np.array([[p['fx'], 0, p['cx']], [0, p['fy'], p['cy']], [0, 0, 1.]], dtype=float)
    D = np.array([d[key] for key in ('k1', 'k2', 'p1', 'p2', 'k3')], dtype=float)
    return K, D


def project(xy, K, D):
    x, y = xy.T
    r2 = x*x + y*y
    k1, k2, p1, p2 = D[:4]
    radial = 1 + k1*r2 + k2*r2*r2
    if len(D) == 5:
        radial += D[4]*r2**3
    return np.column_stack((
        K[0, 0]*(x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)) + K[0, 2],
        K[1, 1]*(y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y) + K[1, 2]))


def unproject(pixels, K, D):
    points = np.ascontiguousarray(pixels[:, None, :], dtype=float)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-13)
    # OpenCV 5 merged the iterative entry point into undistortPoints.
    if hasattr(cv2, 'undistortPointsIter'):
        rays = cv2.undistortPointsIter(points, K, D, None, None, criteria)
    else:
        rays = cv2.undistortPoints(points, K, D, criteria=criteria)
    rays = rays.reshape(-1, 2)
    if np.max(np.linalg.norm(project(rays, K, D) - pixels, axis=1)) > 1e-7:
        raise ValueError('Source distortion inversion did not converge')
    return rays


def unpack(parameters):
    fx, fy, cx, cy = parameters[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]), parameters[4:]


def error_metrics(pixels, reference_rays, K, D, width, height):
    error = np.linalg.norm(project(reference_rays, K, D) - pixels, axis=1)
    border = ((pixels[:, 0] < .1*(width-1)) | (pixels[:, 0] > .9*(width-1))
              | (pixels[:, 1] < .1*(height-1)) | (pixels[:, 1] > .9*(height-1)))
    worst = int(np.argmax(error))
    return dict(rms_px=float(np.sqrt(np.mean(error**2))),
                median_px=float(np.median(error)), p95_px=float(np.quantile(error, .95)),
                p99_px=float(np.quantile(error, .99)), max_px=float(error[worst]),
                worst_pixel=pixels[worst].tolist(),
                outer_10_percent_border_rms_px=float(np.sqrt(np.mean(error[border]**2))),
                outer_10_percent_border_max_px=float(np.max(error[border])))


def fit_camera(camera):
    K, D = source_parameters(camera)
    width, height = int(camera['image_width']), int(camera['image_height'])
    xx, yy = np.meshgrid(np.linspace(0, width-1, 121), np.linspace(0, height-1, 101))
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    rays = unproject(pixels, K, D)
    baseline = project(rays, K, np.zeros(4))
    A = np.stack([(project(rays, K, np.eye(4)[j]) - baseline).ravel() for j in range(4)], axis=1)
    fixed = np.linalg.lstsq(A, (pixels-baseline).ravel(), rcond=None)[0]

    def residual(parameters):
        Q, F = unpack(parameters)
        return (project(rays, Q, F) - pixels).ravel()

    result = least_squares(residual, [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *fixed],
                           x_scale='jac', ftol=1e-13, xtol=1e-13, gtol=1e-10, max_nfev=100)
    fitted_K, fitted_D = unpack(result.x)
    if not result.success or not np.isfinite(result.x).all() or np.any(np.diag(fitted_K)[:2] <= 0):
        raise ValueError('radtan4 approximation failed')
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    validation_pixels = np.column_stack((xx.ravel(), yy.ravel()))
    validation_rays = unproject(validation_pixels, K, D)
    candidates = {}
    for name, Q, F in [('drop_k3', K, D[:4]), ('fixed_intrinsics', K, fixed),
                       ('refit_intrinsics', fitted_K, fitted_D)]:
        candidates[name] = dict(K=Q.tolist(), D=F.tolist(),
            error=error_metrics(validation_pixels, validation_rays, Q, F, width, height))
    return dict(source_K=K.tolist(), source_D=D.tolist(), resolution=[width, height],
                training_grid=[121, 101], validation_pixel_count=width*height,
                objective='Unweighted least squares of forward 2D pixel projection residuals over a uniform pixel grid',
                validation='Every integer pixel coordinate in the original full image; includes all borders and corners',
                coefficient_order=['k1', 'k2', 'p1', 'p2'], selected='refit_intrinsics',
                fit_version=1, optimizer_success=bool(result.success), candidates=candidates)
