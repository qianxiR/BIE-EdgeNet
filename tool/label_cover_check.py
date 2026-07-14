import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, shape
from shapely.affinity import translate
import warnings

warnings.filterwarnings('ignore')


class ChangeLabelFilter:
    def __init__(self, vector_path, raster_path, output_path):
        """
        初始化标签筛选器
        :param vector_path: 精细标签矢量文件路径（shp/geojson）
        :param raster_path: 模型预测栅格文件路径（tif）
        :param output_path: 筛选后矢量文件输出路径
        """
        self.vector_path = vector_path
        self.raster_path = raster_path
        self.output_path = output_path

        # 分层阈值配置（可根据你的数据调整）
        self.area_thresholds = {
            "small": {"pixel_num": 100, "ratio": 0.10},  # 小区域：<100像素，占比≥10%保留
            "medium": {"pixel_num": 1000, "ratio": 0.20},  # 中区域：100-1000像素，占比≥20%保留
            "large": {"pixel_num": float('inf'), "ratio": 0.25}  # 大区域：>1000像素，占比≥25%保留
        }

        # 加载数据
        self.gdf = self._load_vector()
        self.src, self.raster_data = self._load_raster()
        self.pixel_size = self.src.res[0]  # 像素分辨率（米/像素）

    def _load_vector(self):
        """加载矢量数据并检查有效性"""
        gdf = gpd.read_file(self.vector_path)
        print(f"原始矢量数据加载完成，共{len(gdf)}个多边形")

        # 移除空几何
        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf[gdf.geometry.is_valid]
        print(f"过滤无效几何后，剩余{len(gdf)}个多边形")

        return gdf

    def _load_raster(self):
        """加载栅格数据并检查有效性"""
        src = rasterio.open(self.raster_path)
        raster_data = src.read(1)  # 读取第一波段（二值：1=变化，0=无变化）

        # 检查栅格是否为二值
        unique_vals = np.unique(raster_data)
        if not set(unique_vals).issubset({0, 1}):
            print("警告：模型预测栅格不是二值图，自动转换为二值（>0=1）")
            raster_data = (raster_data > 0).astype(np.uint8)

        print(f"模型预测栅格加载完成，尺寸：{raster_data.shape}，变化像素数：{np.sum(raster_data)}")
        return src, raster_data

    def _get_polygon_pixel_mask(self, polygon):
        """
        将单个矢量多边形栅格化为掩码（仅覆盖该多边形的区域）
        :param polygon: shapely多边形对象
        :return: 多边形掩码（1=多边形内，0=外）、掩码范围（行/列切片）
        """
        # 获取多边形的地理范围
        minx, miny, maxx, maxy = polygon.bounds

        # 将地理坐标转换为栅格列/行号
        row_start, col_start = self.src.index(minx, maxy)
        row_end, col_end = self.src.index(maxx, miny)

        # 确保不越界
        row_start = max(0, row_start)
        col_start = max(0, col_start)
        row_end = min(self.src.height, row_end)
        col_end = min(self.src.width, col_end)

        if row_start >= row_end or col_start >= col_end:
            return None, None, None

        # 构建局部地理范围的仿射变换
        local_transform = self.src.transform * rasterio.Affine.translation(col_start, row_start)

        # 栅格化多边形（仅在局部范围内）
        try:
            mask = rasterize(
                [(polygon, 1)],
                out_shape=(row_end - row_start, col_end - col_start),
                transform=local_transform,
                fill=0,
                dtype=np.uint8
            )
        except:
            return None, None, None

        return mask, (row_start, row_end), (col_start, col_end)

    def _calculate_overlap_ratio(self, polygon):
        """
        计算单个多边形内模型预测变化像素的占比
        :param polygon: shapely多边形对象
        :return: 总像素数、变化像素数、重合占比
        """
        # 获取多边形掩码和范围
        mask, row_range, col_range = self._get_polygon_pixel_mask(polygon)
        if mask is None:
            return 0, 0, 0.0

        # 提取该多边形范围内的模型预测数据
        row_start, row_end = row_range
        col_start, col_end = col_range
        pred_data = self.raster_data[row_start:row_end, col_start:col_end]

        # 计算多边形总像素数
        total_pixels = np.sum(mask)
        if total_pixels == 0:
            return 0, 0, 0.0

        # 计算多边形内模型预测的变化像素数
        overlap_pixels = np.sum((mask == 1) & (pred_data == 1))

        # 计算占比
        overlap_ratio = overlap_pixels / total_pixels

        return total_pixels, overlap_pixels, overlap_ratio

    def _get_layer_threshold(self, total_pixels):
        """根据多边形像素数获取对应层级的阈值"""
        if total_pixels < self.area_thresholds["small"]["pixel_num"]:
            return "small", self.area_thresholds["small"]["ratio"]
        elif total_pixels < self.area_thresholds["medium"]["pixel_num"]:
            return "medium", self.area_thresholds["medium"]["ratio"]
        else:
            return "large", self.area_thresholds["large"]["ratio"]

    def filter_labels(self):
        """核心筛选逻辑：逐多边形计算占比，按分层阈值筛选"""
        # 存储筛选结果
        keep_list = []
        stats_list = []

        # 遍历所有多边形
        for idx, row in self.gdf.iterrows():
            polygon = row.geometry
            if not polygon.is_valid:
                stats_list.append({"id": idx, "layer": "invalid", "keep": False})
                continue

            # 计算占比
            total_pixels, overlap_pixels, overlap_ratio = self._calculate_overlap_ratio(polygon)

            # 获取对应层级的阈值
            layer, threshold = self._get_layer_threshold(total_pixels)

            # 判断是否保留
            keep = overlap_ratio >= threshold and total_pixels > 0

            # 记录统计信息
            stats = {
                "id": idx,
                "total_pixels": total_pixels,
                "overlap_pixels": overlap_pixels,
                "overlap_ratio": round(overlap_ratio, 4),
                "layer": layer,
                "threshold": threshold,
                "keep": keep
            }
            stats_list.append(stats)

            # 保留符合条件的多边形
            if keep:
                keep_list.append(row)

            # 打印进度
            if idx % 100 == 0:
                print(f"已处理{idx + 1}/{len(self.gdf)}个多边形，当前保留数：{len(keep_list)}")

        # 生成筛选后的矢量数据
        if keep_list:
            keep_gdf = gpd.GeoDataFrame(keep_list, crs=self.gdf.crs)
            # 保存为Shapefile
            keep_gdf.to_file(self.output_path, driver="ESRI Shapefile")
            print(f"\n筛选完成！")
            print(f"原始多边形数：{len(self.gdf)}")
            print(f"保留多边形数：{len(keep_gdf)}")
            print(f"筛选结果已保存至：{self.output_path}")
        else:
            print("⚠️ 没有符合条件的多边形被保留！")
            return

        # 打印分层统计
        self._print_layer_stats(stats_list)
        return keep_gdf, stats_list

    def _print_layer_stats(self, stats_list):
        """打印分层统计信息"""
        print("\n===== 分层统计结果 =====")
        for layer in ["small", "medium", "large", "invalid"]:
            layer_stats = [s for s in stats_list if s["layer"] == layer]
            if not layer_stats:
                continue

            total = len(layer_stats)
            keep = len([s for s in layer_stats if s["keep"]])
            keep_ratio = keep / total if total > 0 else 0

            print(f"{layer.upper()} 区域：")
            print(f"  总数：{total}，保留数：{keep}，保留率：{keep_ratio:.2%}")
            if total > 0:
                avg_ratio = np.mean([s["overlap_ratio"] for s in layer_stats])
                print(f"  平均重合占比：{avg_ratio:.2%}")


if __name__ == "__main__":
    # ===================== 配置参数（修改为你的路径）=====================
    VECTOR_PATH = r"E:\CDCD\CDCDshp.shp"  # 精细标签矢量文件
    RASTER_PATH = r"E:\CDCD\output_pred.tif"  # 模型预测二值栅格
    OUTPUT_PATH = r"E:\CDCD\CDCDshp_check.shp"  # 筛选后输出路径

    # ===================== 执行筛选 =====================
    filter = ChangeLabelFilter(VECTOR_PATH, RASTER_PATH, OUTPUT_PATH)
    keep_gdf, stats = filter.filter_labels()