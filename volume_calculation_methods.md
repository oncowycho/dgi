# 3D 구조물 부피 측정 방식 비교

의료 영상(예: CT, MRI)에서 종양이나 장기 같은 3차원 구조물의 부피를 계산하는 것은 임상적으로 매우 중요합니다. 부피를 계산하는 데 사용될 수 있는 몇 가지 주요 방식들의 개념과 장단점을 비교합니다.

## 1. 복셀 카운팅 (Voxel Counting)

### 개념
가장 직관적인 방식으로, 구조물 내부에 속하는 것으로 식별된 복셀(voxel, 3D 픽셀)들의 부피를 단순히 합산합니다.

*   각 영상 슬라이스에서 구조물의 2D 윤곽선(contour)을 정의합니다.
*   윤곽선 내부에 중심점이 포함되는 복셀들을 식별합니다.
*   각 복셀의 부피 (`가로 크기 × 세로 크기 × 슬라이스 두께`)를 계산합니다.
*   모든 식별된 복셀의 부피를 더하여 구조물의 총 부피를 결정합니다.

### 장점
*   **구현 용이성:** 알고리즘이 간단하여 이해하고 구현하기 쉽습니다.
*   **직관성:** 계산 과정이 명확합니다.
*   **강건함(Robustness):** 복잡한 형태나 노이즈에 비교적 덜 민감하며, 위상 오류 발생 가능성이 낮습니다.
*   **계산 속도:** 마스크(mask)가 준비되면 계산 속도가 빠를 수 있습니다.

### 단점
*   **계단 현상(Blocky Artifacts):** 결과 표면이 복셀 단위로 각져 보이며 매끄럽지 않습니다.
*   **부분 용적 효과(Partial Volume Effect):** 구조물 경계에 걸쳐 있는 복셀을 이진적(포함 또는 미포함)으로 처리하므로, 특히 해상도가 낮을 때 정확도가 떨어질 수 있습니다.
*  <span style="color:blue;">**표면적 계산 부적합:**</span> 표면적 계산에는 정확도가 떨어집니다.
*   **해상도 의존성:** 복셀 크기에 따라 정확도가 크게 영향을 받습니다.

*현재 `cal_dvh.py`에서 DVH 계산 시 사용하는 방식입니다.*

### 코드 예시 (Python)
```python
import numpy as np

def calculate_volume_voxel_counting(structure_mask, voxel_dims):
    \"\"\"
    복셀 카운팅 방식으로 부피를 계산합니다.

    Args:
        structure_mask (np.ndarray): 구조물 내부 복셀은 True, 외부는 False인 3D boolean 배열.
        voxel_dims (tuple): 복셀의 각 차원 크기 (예: (dx, dy, dz) in mm).

    Returns:
        float: 계산된 부피 (mm³).
    \"\"\"
    if not isinstance(structure_mask, np.ndarray) or structure_mask.ndim != 3:
        raise ValueError("structure_mask는 3차원 numpy 배열이어야 합니다.")
    if not isinstance(voxel_dims, (list, tuple)) or len(voxel_dims) != 3:
        raise ValueError("voxel_dims는 3개의 요소(dx, dy, dz)를 가진 리스트 또는 튜플이어야 합니다.")

    voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
    num_voxels_in_structure = np.sum(structure_mask) # True인 복셀 개수 세기
    total_volume = num_voxels_in_structure * voxel_volume

    return total_volume

# --- 예시 사용법 ---
# 예시 마스크 (5x5x5 크기, 중앙 3x3x3이 구조물)
mask = np.zeros((5, 5, 5), dtype=bool)
mask[1:4, 1:4, 1:4] = True

# 예시 복셀 크기 (1mm x 1mm x 2mm)
dims = (1.0, 1.0, 2.0)

volume = calculate_volume_voxel_counting(mask, dims)
print(f"복셀 카운팅 부피: {volume} mm³") # 예상 결과: 27 * (1*1*2) = 54.0
```

### 복셀 카운팅 실패 사례

복셀 카운팅을 통한 부피 추정은 임계값(threshold) 이상인 복셀(3차원 화소)의 개수를 세고, 해당 복셀 하나의 부피(복셀 크기)를 곱하는 방식입니다.  
그러나 물체의 크기에 비해 복셀 격자가 너무 거칠거나, 물체의 경계가 불규칙하거나 얇은 경우 **부분 부피 효과(partial volume effect)** 가 발생합니다.  
이 경우, 물체가 복셀 내부를 부분적으로만 차지하므로 단순 이진(존재/미존재) 분류로는 실제 부피가 제대로 반영되지 않습니다.  
- 경계가 충분히 샘플링되지 않으면 실제 부피보다 적게 추정되거나  
- 노이즈에 의해 잘못된 복셀이 포함되어 과대 추정될 수도 있습니다.

- **그림 구성:**  
  3차원 불규칙(유기체 형태) 물체(예: 종양 또는 잎사귀 모양) 위에 복셀 격자가 겹쳐진 개략도.
- **핵심 특징:**  
  - 경계에 위치한 일부 복셀은 물체를 부분적으로만 포함하도록 반쯤 회색으로 표시되어 있음.  
  - 확대된 인셋에서는 한 복셀이 물체 경계에 걸쳐 있는 모습을 보여주며, 점선으로 물체가 차지하는 비율을 표시함.
- **캡션:**  
  **"거친 복셀 격자가 불규칙한 물체 위에 겹쳐진 모습. 부분 부피 효과로 인해 경계 복셀이 오분류되어 부피 추정에 오류가 발생한다."**


## 2. 마칭 큐브 (Marching Cubes)

### 개념
복셀 데이터를 기반으로 등가면(isosurface)을 추출하여 3차원 표면 메쉬(mesh)를 생성하는 알고리즘입니다. 특정 값(isovalue, 예: 특정 선량 값)을 기준으로 표면을 정의합니다.

*   데이터 그리드를 순회하며 각 복셀 꼭짓점의 값이 기준값(threshold)보다 큰지 작은지를 확인합니다.
*   미리 정의된 15가지 큐브 패턴에 따라 해당 복셀 내부에 표면을 나타내는 삼각형(triangle)들을 생성합니다.
*   생성된 삼각형들을 연결하여 전체 구조물의 3D 표면 메쉬를 만듭니다.
*   이 메쉬로부터 부피(`skimage.measure.mesh_volume` 등)와 표면적(`skimage.measure.mesh_surface_area` 등)을 계산할 수 있습니다.

### 장점
*   **매끄러운 표면 생성:** 시각화에 적합한 부드러운 3D 표면을 제공합니다.
*   **표면적 계산 가능:** 생성된 메쉬를 이용해 복셀 카운팅보다 잠재적으로 더 정확한 표면적 계산이 가능합니다.
*   **복셀 이하 정밀도(Sub-voxel Precision):** 복셀 사이를 보간하므로 이론적으로 더 미세한 경계 표현이 가능합니다.
*   **표준 알고리즘:** 컴퓨터 그래픽스 및 의료 영상 분야에서 널리 사용됩니다.

### 단점
*   **구현 복잡성:** 복셀 카운팅보다 구현이 복잡합니다.
*   **파라미터 민감성:** 기준값(isovalue) 및 그리드 해상도에 따라 결과가 달라질 수 있습니다.
*   **위상학적 문제:** 복잡한 형태나 노이즈 데이터에서 잘못된 메쉬(예: non-manifold)나 위상 오류가 발생할 수 있습니다.
*   **계산 비용:** 고해상도 데이터에서는 복셀 카운팅보다 느릴 수 있습니다.
*   **부피 정확도 보장 어려움:** 표면은 부드럽지만, 이로부터 계산된 부피가 복셀 카운팅보다 항상 더 정확하다고 단정하기는 어렵습니다.

*현재 `isodose_web_server.py`에서 Isodose 테이블 부피/표면적 계산 시 시도하는 방식입니다 (실패 시 복셀 카운팅으로 대체).*

### 코드 예시 (Python)
```python
import numpy as np
try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image 라이브러리가 설치되지 않았습니다. Marching Cubes 예제를 실행할 수 없습니다.")

def calculate_volume_marching_cubes(data_grid, level, spacing):
    \"\"\"
    Marching Cubes 알고리즘으로 메쉬를 생성하고 부피를 계산합니다.

    Args:
        data_grid (np.ndarray): 3D 데이터 배열 (예: 선량 그리드).
        level (float): 등가면을 정의하는 기준값 (isovalue).
        spacing (tuple): 복셀 간격 (일반적으로 dz, dy, dx 순서).

    Returns:
        float: 계산된 부피 (mm³). 오류 발생 시 None 반환.
    \"\"\"
    if not SKIMAGE_AVAILABLE:
        print("scikit-image가 필요합니다.")
        return None
    if not isinstance(data_grid, np.ndarray) or data_grid.ndim != 3:
        raise ValueError("data_grid는 3차원 numpy 배열이어야 합니다.")
    if not isinstance(spacing, (list, tuple)) or len(spacing) != 3:
        raise ValueError("spacing은 3개의 요소(dz, dy, dx)를 가진 리스트 또는 튜플이어야 합니다.")

    try:
        # Marching cubes 실행 (level 기준으로 표면 추출)
        verts, faces, normals, values = measure.marching_cubes(
            data_grid,
            level=level,
            spacing=spacing # (dz, dy, dx) 순서 주의
        )

        # 생성된 메쉬로부터 부피 계산
        volume = measure.mesh_volume(verts, faces)
        # mesh_volume은 signed volume을 반환할 수 있으므로 절대값 사용
        return abs(volume)

    except Exception as e:
        print(f"Marching Cubes 부피 계산 중 오류 발생: {e}")
        return None

# --- 예시 사용법 ---
if SKIMAGE_AVAILABLE:
    # 예시 데이터 그리드 (구 형태)
    shape = (50, 50, 50)
    center = np.array(shape) / 2
    z, y, x = np.indices(shape)
    distance = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
    data = np.clip(20 - distance, 0, 20) # 중심에서 20, 멀어질수록 감소

    # 예시 복셀 간격 (2mm x 1mm x 1mm)
    # spacing 순서: Z, Y, X
    voxel_spacing = (2.0, 1.0, 1.0)

    # 등가면 기준값 (예: 10)
    isolevel = 10.0

    volume = calculate_volume_marching_cubes(data, isolevel, voxel_spacing)

    if volume is not None:
        print(f"Marching Cubes 부피 (level={isolevel}): {volume:.2f} mm³")
        # 이론적인 구 부피 (반지름=10): (4/3)*pi*10^3 = 4188.79
        # 실제 복셀 간격이 다르므로 위 값과 직접 비교는 어려움.
```
**참고:** `skimage.measure` 라이브러리 설치가 필요합니다 (`pip install scikit-image`).

### 마칭 큐브 실패 사례

마칭 큐브 알고리즘은 부피 데이터를 큐브 단위로 처리하여 등가면(isosurface)을 추출합니다.  
각 큐브에서, 등가면이 어느 모서리를 통과하는지를 결정한 후 삼각분할(triangulation)을 수행합니다.  
그러나, 8개 꼭짓점 중 4개만 임계값 이상이고 4개가 미만인 경우 등과 같이 애매한 경우에는,  
lookup table에 여러 가지 유효한 삼각분할이 존재할 수 있어 모호성이 발생합니다.  
또한, 노이즈나 스칼라 필드의 급격한 변화로 인한 불일치로 인접 큐브 간 삼각분할이 일관되지 않으면  
메시(mesh)에 구멍이나 단절, 또는 "도넛 모양" 오류가 발생할 수 있습니다.

- **그림 구성:**  
  부피 격자 중 하나의 큐브를 클로즈업하여, 여덟 꼭짓점을 표시함.
- **핵심 특징:**  
  - 꼭짓점은 색상으로 구분됨 (임계값 이상은 빨강, 미만은 파랑).  
  - 애매한 경우를 강조하기 위해 두 가지 가능한 삼각분할 예시를 함께 표시 (하나는 구멍이 생긴 경우, 다른 하나는 잘못 연결된 경우).  
  - 인접 큐브 사이의 불일치를 나타내는 화살표나 주석을 포함함.
- **캡션:**  
  **"마칭 큐브 알고리즘에서 모호한 큐브 구성. 보간 불일치로 인해 추출된 등가면에 구멍이나 틈이 발생할 수 있다."**



## 3. 컨벡스 헐 (Convex Hull)

### 개념
주어진 점 집합(예: 구조물을 구성하는 모든 복셀의 중심점 또는 윤곽선 위의 점들)을 모두 포함하는 가장 작은 볼록 다면체(convex polyhedron)를 찾는 방법입니다. 고무줄로 모든 점을 감쌌을 때 만들어지는 형태를 상상하면 이해하기 쉽습니다.

### 장점
*   **개념적 단순성:** "가장 바깥쪽 경계"라는 개념은 직관적입니다.
*   **강건함(Robustness):** 알고리즘 자체는 안정적입니다.
*   **특정 지표에 유용:** 구조물의 전체적인 범위(extent)나 "경계 부피(bounding volume)"를 파악하는 데는 유용할 수 있습니다.
*   **유일성:** 주어진 점 집합에 대해 Convex Hull은 유일하게 정의됩니다.

### 단점
*   **심각한 부피 과대평가 (치명적 단점):** 구조물의 **오목한 부분(concavities)을 완전히 무시**합니다. 대부분의 해부학적 구조(장기, 종양 등)는 오목한 부분을 가지고 있으므로, Convex Hull로 계산된 부피는 **항상 실제 부피보다 크거나 같습니다.** 따라서 **정확한 부피 측정에는 거의 사용되지 않습니다.**
*   **형태 정보 손실:** 내부 디테일이나 오목한 특징에 대한 정보가 완전히 사라집니다.
*   **실제 부피 비대표:** 실제 조직이나 종양의 부피를 정량화하는 목적에는 매우 부적합합니다.

### 코드 예시 (Python)
```python
import numpy as np
try:
    from scipy.spatial import ConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy 라이브러리가 설치되지 않았습니다. Convex Hull 예제를 실행할 수 없습니다.")

def get_structure_points(structure_mask, voxel_dims=(1,1,1), origin=(0,0,0)):
    \"\"\" 구조물 마스크에서 복셀 중심점 좌표를 추출합니다. \"\"\"
    if not isinstance(structure_mask, np.ndarray) or structure_mask.ndim != 3:
        raise ValueError("structure_mask는 3차원 numpy 배열이어야 합니다.")
    
    indices = np.argwhere(structure_mask) # True인 복셀의 인덱스 (z, y, x)
    
    # 인덱스를 실제 좌표로 변환
    points = indices * np.array([voxel_dims[2], voxel_dims[1], voxel_dims[0]]) + np.array([origin[2], origin[1], origin[0]])
    # Scipy는 (x, y, z) 순서를 선호하므로 열 순서 변경
    points = points[:, [2, 1, 0]] 
    return points

def calculate_volume_convex_hull(points):
    \"\"\"
    주어진 3D 점들의 Convex Hull 부피를 계산합니다.

    Args:
        points (np.ndarray): N x 3 형태의 점 좌표 배열.

    Returns:
        float: Convex Hull의 부피. 점이 부족하면 None 반환.
    \"\"\"
    if not SCIPY_AVAILABLE:
        print("scipy가 필요합니다.")
        return None
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points는 N x 3 형태의 numpy 배열이어야 합니다.")
    if points.shape[0] < 4:
        print("Convex Hull을 계산하기에 점이 부족합니다 (최소 4개 필요).")
        return None

    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception as e:
        # QHull 에러 (예: 모든 점이 동일 평면에 있을 때) 처리
        print(f"Convex Hull 부피 계산 중 오류 발생: {e}")
        return None

# --- 예시 사용법 ---
if SCIPY_AVAILABLE:
    # 예시 마스크 (5x5x5 크기, 오목한 부분 없이 꽉 찬 정육면체)
    mask_convex = np.ones((3, 3, 3), dtype=bool) 
    points_convex = get_structure_points(mask_convex, voxel_dims=(1,1,1))
    
    # 예시 마스크 (오목한 부분이 있는 형태, 예: 'ㄷ'자 형태)
    mask_concave = np.ones((5, 5, 5), dtype=bool)
    mask_concave[1:4, 1:4, 1:4] = False # 중앙 부분을 제거하여 오목하게 만듦
    points_concave = get_structure_points(mask_concave, voxel_dims=(1,1,1))

    volume_convex = calculate_volume_convex_hull(points_convex)
    volume_concave_hull = calculate_volume_convex_hull(points_concave)
    actual_concave_volume = np.sum(mask_concave) * 1.0 # 실제 복셀 카운팅 부피

    if volume_convex is not None:
        print(f"Convex 구조물의 Convex Hull 부피: {volume_convex:.2f} mm³")
        # 실제 복셀 부피: 3*3*3 = 27. Convex Hull도 비슷하게 나옴.

    if volume_concave_hull is not None:
        print(f"Concave 구조물의 Convex Hull 부피: {volume_concave_hull:.2f} mm³")
        print(f"Concave 구조물의 실제 복셀 부피: {actual_concave_volume:.2f} mm³")
        # Convex Hull 부피가 실제 복셀 부피보다 훨씬 크게 나옴 (오목한 부분이 채워지므로).
```
**참고:** `scipy` 라이브러리 설치가 필요합니다 (`pip install scipy`). 또한 Convex Hull 방식은 **오목한 구조물의 실제 부피를 심각하게 과대평가**한다는 점을 유념해야 합니다.

### 볼록 껍질 실패 사례

볼록 껍질을 이용한 부피 추정은 물체가 볼록(convex)하다는 전제 하에, 물체를 모두 포함하는 최소 볼록 집합의 부피를 계산합니다.  
그러나 실제 물체가 오목(비볼록)한 경우, 볼록 껍질은 물체의 오목한 영역을 “채워” 버려 실제 부피보다 크게 추정됩니다.  
이 문제는 생물학적 형태나 공학적 부품에서, 인덱이나 홈 등 오목한 특징을 가진 물체에서 자주 발생합니다.

### 구체적 삽화 설명
- **그림 구성:**  
  2차원 스케치(3차원으로 확장 가능)로, 초승달 모양 또는 U자형의 오목한 물체를 표현함.
- **핵심 특징:**  
  - 실제 물체의 오목한 경계가 명확히 그려짐.  
  - 볼록 껍질은 매끄러운 점선이나 실선으로 오목 부분을 “채워” 물체의 전체 영역을 덮어버림.  
  - 볼록 껍질과 실제 물체 경계 사이의 차이가 주석이나 음영으로 강조됨.
- **캡션:**  
  **"오목한 물체와 그 볼록 껍질 오버레이. 볼록 껍질 방식은 오목 부분을 채워 실제 부피를 과대 추정한다."**


### 복셀 카운팅 vs 마칭 큐브 방식의 부피 계산 차이
## 1. 복셀 카운팅 방식 (Voxel Counting)
- 정의: 바이너리 마스크 혹은 세그멘테이션 결과에서 1로 라벨링된 복셀의 개수를 세고, 복셀당 실제 부피(픽셀 사이즈)를 곱해서 총 부피를 계산합니다.
- 특징: 매우 단순하고 직관적이며, 격자 기반이기 때문에 계산이 빠릅니다.
- 한계: 경계가 계단식으로 처리되어 실제 표면보다 과소 혹은 과대 추정 가능성이 있습니다.

## 2. 마칭 큐브 방식 (Marching Cubes)
- 정의: 3D 바이너리 볼륨에서 등치선(isosurface)을 추정하여 삼각형 메시를 구성하고, 이 메시를 기반으로 부피를 계산합니다.
- 특징: 경계를 더 부드럽게 모델링하여 복셀 기반보다 실제 모양에 가까운 표면을 생성합니다.
- 장점: 더 정확한 표면 근사 → 실제 부피를 더 잘 반영함.

- 단점: 복잡도와 계산 비용이 더 큼.

# 부피 차이: 어느 정도?
차이의 크기는 다음 요소들에 따라 달라집니다:
- 복셀 크기 (해상도): 해상도가 낮을수록 복셀 방식은 경계에서 부정확해짐.
- 형상의 복잡도: 표면이 매끄럽고 곡면이 많을수록 마칭 큐브의 정확도가 복셀보다 현저히 높습니다.
- 스무딩 여부: 마칭 큐브 적용 전후에 smoothing 또는 interpolation을 했는지 여부.

# 논문 및 실험 예시:
보통 실험 결과에 따르면 5~15% 정도의 차이가 관찰되며, 복셀 방식이 약간 과소 추정하거나 과대 추정하는 경향이 있습니다.

예를 들어, 복셀 방식으로 1000 mm³로 나온 부피가 마칭 큐브 기반 메시 계산에서는 1070 mm³이 나올 수 있습니다.

# 결론
단순한 구조에서는 두 방식의 부피 차이는 크지 않지만 (5% 이내),

복잡하고 경계가 부드러운 형태일수록 마칭 큐브 방식이 더 정확한 부피를 제공합니다.

연구나 임상에서 정밀한 부피 추정이 중요할 경우, 마칭 큐브 + 메시 기반 계산이 더 권장됩니다.

## 요약

*   **Voxel Counting:** 간단하고 빠르지만 결과가 거칠 수 있습니다. DVH 계산에 주로 사용됩니다.
*   **Marching Cubes:** 매끄러운 표면을 생성하여 시각화 및 표면적 계산에 유리하지만, 구현이 복잡하고 오류 가능성이 있습니다. Isodose 시각화 및 관련 메트릭 계산에 사용될 수 있습니다.
*   **Convex Hull:** 구조물의 범위를 파악하는 데는 유용할 수 있으나, 오목한 부분을 무시하여 실제 부피를 심각하게 과대평가하므로 정확한 부피 측정에는 사용하지 않는 것이 좋습니다. 
