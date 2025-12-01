"""
image_enhance.py - 图像增强与质量评估

包含:
1. 失真分析 - 亮度/对比度/噪声/光照均匀性检测
2. 增强方法 - 直方图均衡, CLAHE, 对比度拉伸, 锐化
3. 去噪方法 - 高斯/双边/中值/NLM滤波
4. 质量评估 - BRISQUE无参考评分
"""
import os
import cv2
import numpy as np
import torch

try:
    from skimage import img_as_float, exposure
    from skimage.restoration import estimate_sigma
    from skimage import util
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

try:
    from brisque import BRISQUE
    BRISQUE_AVAILABLE = True
except Exception:
    BRISQUE_AVAILABLE = False

try:
    import piq
    PIQ_AVAILABLE = True
except Exception:
    PIQ_AVAILABLE = False


def to_gray(img):
    """BGR转灰度"""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ============================================================
#                       失真分析
# ============================================================

def analyze_brightness(img):
    """分析亮度，返回均值和问题类型"""
    gray = to_gray(img)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 60:
        problem = 'too_dark'  # 过暗
        severity = (60 - mean_brightness) / 60
    elif mean_brightness > 200:
        problem = 'too_bright'  # 过亮/过曝
        severity = (mean_brightness - 200) / 55
    else:
        problem = 'normal'
        severity = 0
    
    return {
        'mean': round(float(mean_brightness), 2),
        'problem': problem,
        'severity': round(min(1.0, severity), 2)
    }


def analyze_contrast(img):
    """分析对比度，返回标准差和问题类型"""
    gray = to_gray(img)
    contrast = np.std(gray)
    
    # 计算动态范围
    min_val, max_val = np.min(gray), np.max(gray)
    dynamic_range = max_val - min_val
    
    if contrast < 25:
        problem = 'low_contrast'  # 对比度低
        severity = (25 - contrast) / 25
    elif dynamic_range < 100:
        problem = 'narrow_range'  # 动态范围窄
        severity = (100 - dynamic_range) / 100
    else:
        problem = 'normal'
        severity = 0
    
    return {
        'std': round(float(contrast), 2),
        'dynamic_range': int(dynamic_range),
        'problem': problem,
        'severity': round(min(1.0, severity), 2)
    }


def analyze_noise(img):
    """分析噪声/模糊程度，用拉普拉斯方差估计"""
    gray = to_gray(img)
    
    # 方法1：拉普拉斯方差（高=清晰或噪声大，低=模糊）
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = lap.var()
    
    # 方法2：使用 skimage 估计噪声（如果可用）
    noise_sigma = None
    if SKIMAGE_AVAILABLE:
        try:
            noise_sigma = estimate_sigma(gray)
        except:
            pass
    
    # 判断噪声问题
    if noise_sigma is not None and noise_sigma > 10:
        problem = 'noisy'
        severity = min(1.0, noise_sigma / 30)
    elif lap_var < 50:
        problem = 'blurry'  # 模糊
        severity = (50 - lap_var) / 50
    else:
        problem = 'normal'
        severity = 0
    
    result = {
        'laplacian_var': round(float(lap_var), 2),
        'problem': problem,
        'severity': round(min(1.0, severity), 2)
    }
    if noise_sigma is not None:
        result['noise_sigma'] = round(float(noise_sigma), 2)
    
    return result


def analyze_uniformity(img):
    """分析光照均匀性，检测局部过亮/过暗"""
    gray = to_gray(img)
    
    # 将图像分成 4x4 的区块
    h, w = gray.shape
    block_h, block_w = h // 4, w // 4
    
    block_means = []
    for i in range(4):
        for j in range(4):
            block = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            block_means.append(np.mean(block))
    
    block_means = np.array(block_means)
    mean_of_means = np.mean(block_means)
    std_of_means = np.std(block_means)
    max_diff = np.max(block_means) - np.min(block_means)
    
    # 判断是否有不均匀光照
    if std_of_means > 30 or max_diff > 80:
        problem = 'uneven_lighting'  # 光照不均
        severity = min(1.0, std_of_means / 50)
    else:
        problem = 'normal'
        severity = 0
    
    return {
        'block_std': round(float(std_of_means), 2),
        'max_diff': round(float(max_diff), 2),
        'problem': problem,
        'severity': round(min(1.0, severity), 2)
    }


def analyze_distortion(img):
    """综合失真分析，返回各项指标和推荐方法"""
    brightness = analyze_brightness(img)
    contrast = analyze_contrast(img)
    noise = analyze_noise(img)
    uniformity = analyze_uniformity(img)
    
    # 收集所有问题
    problems = []
    if brightness['problem'] != 'normal':
        problems.append(('brightness', brightness['problem'], brightness['severity']))
    if contrast['problem'] != 'normal':
        problems.append(('contrast', contrast['problem'], contrast['severity']))
    if noise['problem'] != 'normal':
        problems.append(('noise', noise['problem'], noise['severity']))
    if uniformity['problem'] != 'normal':
        problems.append(('uniformity', uniformity['problem'], uniformity['severity']))
    
    # 根据问题推荐增强方法
    recommendations = []
    
    # 按严重程度排序
    problems.sort(key=lambda x: x[2], reverse=True)
    
    for category, problem, severity in problems:
        if problem == 'too_dark':
            recommendations.append(('clahe', '图像过暗，建议使用 CLAHE 提升亮度和对比度'))
        elif problem == 'too_bright':
            recommendations.append(('stretch', '图像过亮，建议使用对比度拉伸'))
        elif problem == 'low_contrast':
            recommendations.append(('clahe', '对比度低，建议使用 CLAHE'))
        elif problem == 'narrow_range':
            recommendations.append(('stretch', '动态范围窄，建议使用对比度拉伸'))
        elif problem == 'noisy':
            recommendations.append(('denoise_bilateral', '噪声较大，建议使用双边滤波去噪'))
        elif problem == 'blurry':
            recommendations.append(('unsharp', '图像模糊，建议使用反锐化掩模增强'))
        elif problem == 'uneven_lighting':
            recommendations.append(('clahe', '光照不均，建议使用 CLAHE 自适应均衡'))
    
    # 如果没有明显问题，默认推荐 CLAHE
    if not recommendations:
        recommendations.append(('clahe', '图像质量正常，可使用 CLAHE 进一步增强'))
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'noise': noise,
        'uniformity': uniformity,
        'problems': problems,
        'recommendations': recommendations
    }


# ============================================================
#                       去噪
# ============================================================

def gaussian_denoise(img, kernel_size=5):
    """高斯滤波"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def bilateral_denoise(img, d=9, sigma_color=75, sigma_space=75):
    """双边滤波，保边去噪"""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def median_denoise(img, kernel_size=5):
    """中值滤波，去椒盐噪声"""
    return cv2.medianBlur(img, kernel_size)


def nlm_denoise(img, h=10, template_size=7, search_size=21):
    """NLM非局部均值去噪，效果好但慢"""
    if len(img.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, template_size, search_size)
    else:
        return cv2.fastNlMeansDenoising(img, None, h, template_size, search_size)


def denoise_image(img, method='bilateral'):
    """去噪接口"""
    if method == 'gaussian':
        return gaussian_denoise(img)
    elif method == 'bilateral':
        return bilateral_denoise(img)
    elif method == 'median':
        return median_denoise(img)
    elif method == 'nlm':
        return nlm_denoise(img)
    else:
        raise ValueError(f'Unknown denoise method: {method}')


# ============================================================
#                       增强
# ============================================================

def hist_equalization(img):
    """直方图均衡化"""
    gray = to_gray(img)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR) if len(img.shape) == 3 else eq


def clahe_equalization(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE自适应直方图均衡"""
    gray = to_gray(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(gray)
    return cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR) if len(img.shape) == 3 else cl


def contrast_stretch(img, in_range=(2,98)):
    """对比度拉伸，基于百分位数"""
    if SKIMAGE_AVAILABLE:
        imgf = img_as_float(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if len(img.shape)==3 else img_as_float(img)
        p_low, p_high = np.percentile(imgf, in_range)
        out = exposure.rescale_intensity(imgf, in_range=(p_low, p_high))
        out = (out * 255).astype(np.uint8)
        if len(img.shape) == 3:
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out
    else:
        # fallback: simple linear stretch based on percentiles in BGR
        out = img.copy().astype(np.float32)
        for c in range(3 if len(img.shape)==3 else 1):
            channel = out[:,:,c] if len(img.shape)==3 else out
            lo = np.percentile(channel, in_range[0])
            hi = np.percentile(channel, in_range[1])
            channel = (channel - lo) * (255.0 / max(1e-6, (hi - lo)))
            channel = np.clip(channel, 0, 255)
            if len(img.shape)==3:
                out[:,:,c] = channel
            else:
                out = channel
        return out.astype(np.uint8)


def unsharp_mask(img, kernel_size=(5,5), sigma=1.0, amount=1.0, threshold=0):
    """USM锐化"""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    blurred = cv2.GaussianBlur(img_gray, kernel_size, sigma)
    sharpened = float(amount + 1) * img_gray - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.abs(img_gray - blurred) < threshold
        np.copyto(sharpened, img_gray, where=low_contrast_mask)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR) if len(img.shape)==3 else sharpened


def enhance_image(img, method='clahe'):
    """增强接口"""
    if method == 'hist':
        return hist_equalization(img)
    elif method == 'clahe':
        return clahe_equalization(img)
    elif method == 'stretch':
        return contrast_stretch(img)
    elif method == 'unsharp':
        return unsharp_mask(img)
    # 去噪方法
    elif method == 'denoise_gaussian':
        return gaussian_denoise(img)
    elif method == 'denoise_bilateral':
        return bilateral_denoise(img)
    elif method == 'denoise_median':
        return median_denoise(img)
    elif method == 'denoise_nlm':
        return nlm_denoise(img)
    else:
        raise ValueError(f'Unknown enhance method: {method}')


def auto_enhance(img):
    """自动分析并增强，返回(结果图, 分析, 方法列表)"""
    analysis = analyze_distortion(img)
    result = img.copy()
    applied_methods = []
    
    # 按推荐顺序应用增强
    for method, reason in analysis['recommendations']:
        if method.startswith('denoise_'):
            # 去噪方法
            result = denoise_image(result, method.replace('denoise_', ''))
        else:
            # 增强方法
            result = enhance_image(result, method)
        applied_methods.append((method, reason))
        
        # 只应用最重要的1-2个方法，避免过度处理
        if len(applied_methods) >= 2:
            break
    
    return result, analysis, applied_methods


# ============================================================
#                       质量评估
# ============================================================

def quality_metric_simple(img):
    """计算对比度(std)和锐度(laplacian var)"""
    gray = to_gray(img)
    contrast = float(np.std(gray))
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap.var())
    return {'contrast': contrast, 'sharpness': sharpness}


def compute_brisque(img):
    """BRISQUE评分，越低越好(0-100)"""
    if not BRISQUE_AVAILABLE:
        return None
    try:
        # brisque 库需要 RGB 图像
        if len(img.shape) == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        obj = BRISQUE(url=False)
        score = obj.score(rgb)
        return float(score)
    except Exception as e:
        print(f"BRISQUE error: {e}")
        return None


def compute_niqe(img):
    """piq库的BRISQUE实现(备用)"""
    if not PIQ_AVAILABLE:
        return None
    try:
        # piq 需要 torch tensor，形状 (N, C, H, W)，值范围 [0, 1]
        if len(img.shape) == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # 转换为 float tensor
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        # 使用 piq 的 brisque 函数（和 brisque 库不同的实现）
        score = piq.brisque(tensor, data_range=1.0)
        return float(score.item())
    except Exception as e:
        print(f"PIQ-BRISQUE error: {e}")
        return None


def evaluate_image(img):
    """综合评估: 对比度, 锐度, BRISQUE"""
    metrics = quality_metric_simple(img)
    
    # 计算 BRISQUE
    brisque_score = compute_brisque(img)
    if brisque_score is not None:
        metrics['brisque'] = round(brisque_score, 2)
    
    return metrics


def print_available_metrics():
    """打印可用方法"""
    print("可用的图像质量评价指标:")
    print(f"  - 对比度 (contrast): ✓")
    print(f"  - 锐度 (sharpness): ✓")
    print(f"  - BRISQUE: {'✓' if BRISQUE_AVAILABLE else '✗ (pip install brisque)'}")
    print("\n可用的增强方法:")
    print("  - hist: 全局直方图均衡化")
    print("  - clahe: CLAHE 自适应直方图均衡")
    print("  - stretch: 对比度拉伸")
    print("  - unsharp: 反锐化掩模（增强边缘）")
    print("\n可用的去噪方法:")
    print("  - denoise_gaussian: 高斯滤波")
    print("  - denoise_bilateral: 双边滤波（保边去噪）")
    print("  - denoise_median: 中值滤波（去椒盐噪声）")
    print("  - denoise_nlm: 非局部均值去噪（效果好但慢）")
    print("\n自动增强:")
    print("  - auto: 自动分析失真类型并应用推荐方法")


if __name__ == '__main__':
    print('image_enhance module - 图像增强与质量评价')
    print_available_metrics()
