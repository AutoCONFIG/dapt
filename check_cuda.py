import subprocess
import sys
import platform
import time
import threading
import queue
import warnings
import re
import os
import shutil
from typing import Dict, List, Tuple, Optional

def print_header(title):
    print(f"\n{title}")
    print("-" * 60)

def print_subheader(title):
    print(f"\n  {title}")
    print("  " + "-" * 40)

class CUDADetector:
    def __init__(self):
        self.gpu_info = {}
        self.driver_info = {}
        self.pytorch_info = {}
        self.tensorflow_info = {}
        self.conda_info = {}

    def detect_system_info(self):
        """检测系统基本信息"""
        print_header("系统信息检测")

        print(f"操作系统: {platform.system()} {platform.release()}")
        print(f"Python版本: {sys.version.split()[0]}")
        print(f"Python路径: {sys.executable}")

        return True

    def _find_conda_executable(self):
        """查找conda可执行文件"""
        # 常见的conda可执行文件路径
        conda_candidates = [
            'conda',
            '/opt/conda/bin/conda',
            '/opt/conda/condabin/conda',
            '/opt/miniconda/bin/conda',
            '/opt/miniconda/condabin/conda',
            os.path.join(os.path.dirname(sys.executable), 'conda'),
            os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'bin', 'conda'),
            os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'condabin', 'conda'),
        ]

        # 检查环境变量
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_candidates.extend([
                os.path.join(conda_prefix, 'bin', 'conda'),
                os.path.join(conda_prefix, 'condabin', 'conda'),
                os.path.join(os.path.dirname(conda_prefix), 'bin', 'conda'),
                os.path.join(os.path.dirname(conda_prefix), 'condabin', 'conda'),
            ])

        # 使用shutil.which检查PATH中的conda
        conda_in_path = shutil.which('conda')
        if conda_in_path:
            conda_candidates.insert(0, conda_in_path)

        # 测试每个候选路径
        for conda_path in conda_candidates:
            if conda_path and os.path.isfile(conda_path) and os.access(conda_path, os.X_OK):
                try:
                    # 测试conda是否可以正常运行
                    result = subprocess.run(
                        [conda_path, '--version'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        return conda_path
                except:
                    continue

        return None

    def _run_conda_command(self, cmd_args, timeout=30):
        """安全地运行conda命令"""
        conda_exe = self._find_conda_executable()
        if not conda_exe:
            return None, "conda可执行文件未找到"

        try:
            full_cmd = [conda_exe] + cmd_args
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result, None
        except subprocess.TimeoutExpired:
            return None, f"命令超时 (>{timeout}s)"
        except Exception as e:
            return None, str(e)

    def detect_nvidia_driver(self):
        """检测NVIDIA驱动信息"""
        print_header("NVIDIA驱动检测")

        try:
            smi_output = subprocess.check_output(
                "nvidia-smi", shell=True, stderr=subprocess.STDOUT
            ).decode()

            # 提取驱动版本
            driver_match = re.search(r"Driver Version:\s*(\d+\.\d+)", smi_output)
            if driver_match:
                driver_version = driver_match.group(1)
                self.driver_info['version'] = driver_version
                print(f"驱动版本: {driver_version}")

            # 提取支持的最高CUDA版本
            cuda_match = re.search(r"CUDA Version:\s*(\d+\.\d+)", smi_output)
            if cuda_match:
                max_cuda = cuda_match.group(1)
                self.driver_info['max_cuda'] = max_cuda
                print(f"支持的最高CUDA版本: {max_cuda}")

            # 提取GPU信息
            gpu_lines = [line for line in smi_output.split('\n') if 'MiB' in line and 'GeForce' in line or 'Tesla' in line or 'Quadro' in line or 'RTX' in line or 'GTX' in line]
            if not gpu_lines:
                # 更通用的GPU检测
                gpu_lines = [line for line in smi_output.split('\n') if re.search(r'\d+MiB\s*/\s*\d+MiB', line)]

            if gpu_lines:
                for i, line in enumerate(gpu_lines):
                    # 提取GPU名称
                    gpu_name_match = re.search(r'(GeForce|Tesla|Quadro|RTX|GTX|A\d+|H\d+|T\d+|V\d+)[\w\s\-]+', line)
                    if gpu_name_match:
                        gpu_name = gpu_name_match.group().strip()
                    else:
                        gpu_name = f"GPU {i}"

                    # 提取显存信息
                    memory_match = re.search(r'(\d+)MiB\s*/\s*(\d+)MiB', line)
                    if memory_match:
                        used_memory = int(memory_match.group(1))
                        total_memory = int(memory_match.group(2))
                        self.gpu_info[f'gpu_{i}'] = {
                            'name': gpu_name,
                            'memory_used': used_memory,
                            'memory_total': total_memory
                        }
                        print(f"GPU {i}: {gpu_name}")
                        print(f"  显存: {used_memory}MB / {total_memory}MB ({total_memory/1024:.1f}GB)")

            return True

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"未检测到NVIDIA驱动或GPU: {e}")
            return False

    def detect_cuda_runtime(self):
        """检测系统CUDA运行时"""
        print_header("CUDA运行时检测")

        # 检查nvcc
        try:
            nvcc_output = subprocess.check_output(
                "nvcc --version", shell=True, stderr=subprocess.STDOUT
            ).decode()

            version_match = re.search(r"release\s*(\d+\.\d+)", nvcc_output)
            if version_match:
                nvcc_version = version_match.group(1)
                print(f"NVCC版本: {nvcc_version}")
                self.driver_info['nvcc_version'] = nvcc_version
            else:
                print("NVCC已安装但无法识别版本")

        except (subprocess.CalledProcessError, FileNotFoundError):
            print("NVCC未安装或不在PATH中，如果无需编译CUDA内核或PyTorch扩展等功能，可以忽略此问题")

        # 检查CUDA库路径
        cuda_paths = [
            "/usr/local/cuda",
            "/opt/cuda",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
            "C:\\CUDA"
        ]

        for path in cuda_paths:
            if os.path.exists(path):
                print(f"检测到CUDA安装路径: {path}")
                break

        return True

    def detect_pytorch_env(self):
        """检测PyTorch环境"""
        print_header("PyTorch环境检测")

        try:
            import torch
            self.pytorch_info['installed'] = True
            self.pytorch_info['version'] = torch.__version__
            print(f"PyTorch版本: {torch.__version__}")

            # CUDA编译信息
            cuda_compiled = torch.version.cuda
            self.pytorch_info['cuda_compiled'] = cuda_compiled
            if cuda_compiled:
                print(f"编译时CUDA版本: {cuda_compiled}")
            else:
                print("编译时CUDA版本: 无 (CPU版本)")
                return False

            # CUDA可用性
            cuda_available = torch.cuda.is_available()
            self.pytorch_info['cuda_available'] = cuda_available
            print(f"CUDA运行时可用: {cuda_available}")

            if cuda_available:
                # 设备信息
                device_count = torch.cuda.device_count()
                self.pytorch_info['device_count'] = device_count
                print(f"可用GPU设备数: {device_count}")

                for i in range(device_count):
                    device_name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)

                    print(f"  设备 {i}: {device_name}")
                    print(f"    计算能力: {props.major}.{props.minor}")
                    print(f"    总显存: {props.total_memory / 1024**3:.1f} GB")
                    print(f"    多处理器数: {props.multi_processor_count}")

                    # 检查计算能力兼容性
                    compute_capability = f"{props.major}{props.minor}"
                    self.check_compute_compatibility(device_name, compute_capability)

                return True
            else:
                print("CUDA不可用，可能原因:")
                print("  - PyTorch为CPU版本")
                print("  - CUDA版本不兼容")
                print("  - 驱动版本过旧")
                return False

        except ImportError:
            print("PyTorch未安装")
            self.pytorch_info['installed'] = False
            return False

    def detect_tensorflow_env(self):
        """检测TensorFlow环境"""
        print_header("TensorFlow环境检测")

        try:
            import tensorflow as tf
            self.tensorflow_info['installed'] = True
            self.tensorflow_info['version'] = tf.__version__
            print(f"TensorFlow版本: {tf.__version__}")

            # 检查CUDA编译信息
            build_info = tf.sysconfig.get_build_info()
            cuda_version = build_info.get('cuda_version', 'N/A')
            cudnn_version = build_info.get('cudnn_version', 'N/A')

            self.tensorflow_info['cuda_compiled'] = cuda_version
            self.tensorflow_info['cudnn_compiled'] = cudnn_version

            if cuda_version != 'N/A':
                print(f"编译时CUDA版本: {cuda_version}")
                print(f"编译时cuDNN版本: {cudnn_version}")
            else:
                print("编译时CUDA版本: 无 (CPU版本)")

            # 检查GPU可用性
            try:
                # TensorFlow 2.x方式
                gpu_devices = tf.config.list_physical_devices('GPU')
                self.tensorflow_info['gpu_available'] = len(gpu_devices) > 0
                self.tensorflow_info['gpu_count'] = len(gpu_devices)

                print(f"GPU设备数量: {len(gpu_devices)}")

                if gpu_devices:
                    print("TensorFlow GPU设备:")
                    for i, device in enumerate(gpu_devices):
                        print(f"  设备 {i}: {device.name}")

                        # 尝试获取设备详细信息
                        try:
                            device_details = tf.config.experimental.get_device_details(device)
                            compute_capability = device_details.get('compute_capability')
                            if compute_capability:
                                print(f"    计算能力: {compute_capability[0]}.{compute_capability[1]}")
                        except:
                            pass

                    # 检查内存增长设置
                    try:
                        for device in gpu_devices:
                            tf.config.experimental.set_memory_growth(device, True)
                        print("GPU内存增长模式: 已启用")
                    except Exception as e:
                        print(f"GPU内存增长设置: {e}")

                    return True
                else:
                    print("TensorFlow未检测到可用GPU")
                    print("可能原因:")
                    print("  - TensorFlow为CPU版本")
                    print("  - CUDA版本不兼容")
                    print("  - cuDNN版本不兼容")
                    return False

            except AttributeError:
                # TensorFlow 1.x fallback
                try:
                    with tf.Session() as sess:
                        devices = sess.list_devices()
                        gpu_devices = [d for d in devices if d.name.lower().find('gpu') >= 0]
                        self.tensorflow_info['gpu_available'] = len(gpu_devices) > 0
                        self.tensorflow_info['gpu_count'] = len(gpu_devices)
                        print(f"GPU设备数量: {len(gpu_devices)}")
                        return len(gpu_devices) > 0
                except:
                    print("无法检测GPU设备")
                    return False

        except ImportError:
            print("TensorFlow未安装")
            self.tensorflow_info['installed'] = False
            return False
        except Exception as e:
            print(f"TensorFlow检测出错: {e}")
            self.tensorflow_info['installed'] = True
            self.tensorflow_info['error'] = str(e)
            return False

    def check_compute_compatibility(self, device_name: str, compute_capability: str):
        """检查计算能力兼容性"""
        print(f"    计算能力兼容性检查:")

        # 常见的计算能力映射
        capability_info = {
            '35': 'Kepler (GTX 700系列)',
            '37': 'Kepler (部分Tesla)',
            '50': 'Maxwell 1.0 (GTX 900系列)',
            '52': 'Maxwell 2.0 (GTX 900系列)',
            '60': 'Pascal (GTX 10系列)',
            '61': 'Pascal (GTX 10系列)',
            '70': 'Volta (Tesla V100)',
            '72': 'Volta (Jetson)',
            '75': 'Turing (RTX 20系列)',
            '80': 'Ampere (RTX 30系列)',
            '86': 'Ampere (RTX 30系列)',
            '87': 'Ampere (Jetson)',
            '90': 'Hopper (H100)',
            '120': 'Ada Lovelace (RTX 40/50系列)'
        }

        arch_name = capability_info.get(compute_capability, f"未知架构 (sm_{compute_capability})")
        print(f"      架构: {arch_name}")

        # 检查PyTorch支持
        if hasattr(self, 'pytorch_info') and self.pytorch_info.get('installed'):
            import torch
            if hasattr(torch.cuda, 'get_arch_list'):
                supported_archs = torch.cuda.get_arch_list()
                arch_supported = any(compute_capability in arch for arch in supported_archs)
                print(f"      PyTorch支持: {'是' if arch_supported else '否'}")
                if not arch_supported:
                    print(f"        支持的架构: {supported_archs}")

    def run_tensorflow_compatibility_test(self):
        """运行TensorFlow兼容性测试"""
        print_header("TensorFlow兼容性测试")

        if not self.tensorflow_info.get('gpu_available', False):
            print("跳过TensorFlow GPU测试: GPU不可用")
            return False

        try:
            import tensorflow as tf
            test_results = {}

            print_subheader("基础功能测试")

            # 测试1: 张量创建和设备分配
            try:
                print("张量创建和设备分配", end=' ... ')
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                print("成功")
                test_results['tensor_creation'] = True
            except Exception as e:
                print(f"失败: {e}")
                test_results['tensor_creation'] = False

            # 测试2: 基础运算
            try:
                print("基础张量运算", end=' ... ')
                with tf.device('/GPU:0'):
                    c = tf.matmul(a, b)
                    d = tf.nn.relu(c)
                print("成功")
                test_results['basic_ops'] = True
            except Exception as e:
                print(f"失败: {e}")
                test_results['basic_ops'] = False

            # 测试3: 矩阵乘法性能测试
            print("矩阵乘法运算", end=' ... ')
            success = self._test_tensorflow_matmul()
            print("成功" if success else "失败")
            test_results['matmul'] = success

            print_subheader("内存管理测试")

            # 测试4: GPU内存管理
            try:
                print("GPU内存管理", end=' ... ')
                with tf.device('/GPU:0'):
                    # 创建较大的张量
                    large_tensor = tf.random.normal([1000, 1000])
                    result = tf.reduce_sum(large_tensor)

                # 手动清理
                del large_tensor, result
                print("成功")
                test_results['memory_management'] = True
            except Exception as e:
                print(f"失败: {e}")
                test_results['memory_management'] = False

            print_subheader("性能基准测试")
            self._run_tensorflow_performance_benchmark()

            # 汇总结果
            passed_tests = sum(test_results.values())
            total_tests = len(test_results)

            print(f"\nTensorFlow测试汇总: {passed_tests}/{total_tests} 项通过")
            return passed_tests >= total_tests * 0.6

        except Exception as e:
            print(f"TensorFlow兼容性测试失败: {e}")
            return False

    def _test_tensorflow_matmul(self, timeout=10):
        """TensorFlow矩阵乘法测试"""
        result_queue = queue.Queue()

        def matmul_worker():
            try:
                import tensorflow as tf
                # 自适应测试大小
                for size in [64, 128, 256, 512]:
                    try:
                        with tf.device('/GPU:0'):
                            a = tf.random.normal([size, size])
                            b = tf.random.normal([size, size])
                            c = tf.matmul(a, b)
                        result_queue.put((True, size))
                        return
                    except:
                        continue
                result_queue.put((False, 0))
            except Exception as e:
                result_queue.put((False, str(e)))

        thread = threading.Thread(target=matmul_worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)

        try:
            return result_queue.get_nowait()[0]
        except queue.Empty:
            return False

    def _run_tensorflow_performance_benchmark(self):
        """运行TensorFlow性能基准测试"""
        try:
            import tensorflow as tf
            import time

            # 找到合适的测试大小
            test_size = 256
            for size in [128, 256, 512, 1024, 2048, 4096, 8192]:
                try:
                    with tf.device('/GPU:0'):
                        test_tensor = tf.random.normal([size, size])
                    del test_tensor
                    test_size = size
                except:
                    break

            print(f"使用测试矩阵大小: {test_size}x{test_size}")

            # CPU基准
            with tf.device('/CPU:0'):
                cpu_tensor_a = tf.random.normal([test_size, test_size])
                cpu_tensor_b = tf.random.normal([test_size, test_size])
                start_time = time.time()
                cpu_result = tf.matmul(cpu_tensor_a, cpu_tensor_b)
                cpu_time = time.time() - start_time

            # GPU基准
            with tf.device('/GPU:0'):
                gpu_tensor_a = tf.random.normal([test_size, test_size])
                gpu_tensor_b = tf.random.normal([test_size, test_size])
                start_time = time.time()
                gpu_result = tf.matmul(gpu_tensor_a, gpu_tensor_b)
                gpu_time = time.time() - start_time

            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"CPU时间: {cpu_time*1000:.2f}ms")
                print(f"GPU时间: {gpu_time*1000:.2f}ms")
                print(f"加速比: {speedup:.2f}x")
            else:
                print("GPU运算极快，无法准确测量")

        except Exception as e:
            print(f"TensorFlow性能测试失败: {e}")

    def run_gpu_compatibility_test(self):
        """运行GPU兼容性测试"""
        print_header("GPU兼容性测试 - PyTorch")

        if not self.pytorch_info.get('cuda_available', False):
            print("跳过PyTorch GPU测试: CUDA不可用")
            pytorch_success = False
        else:
            pytorch_success = self._run_pytorch_tests()

        return pytorch_success

    def _run_pytorch_tests(self):
        """运行PyTorch测试"""
        try:
            import torch
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

            torch.cuda.empty_cache()
            test_results = {}

            print_subheader("基础功能测试")

            # 测试1: 张量创建和转移
            try:
                print("张量创建和设备转移", end=' ... ')
                cpu_tensor = torch.randn(100, 100)
                gpu_tensor = cpu_tensor.cuda()
                back_to_cpu = gpu_tensor.cpu()
                print("成功")
                test_results['tensor_transfer'] = True
            except Exception as e:
                print(f"失败: {e}")
                test_results['tensor_transfer'] = False

            # 测试2: 基础运算
            try:
                print("基础张量运算", end=' ... ')
                a = torch.randn(50, 50, device='cuda')
                b = torch.randn(50, 50, device='cuda')
                c = a + b
                d = torch.relu(c)
                torch.cuda.synchronize()
                print("成功")
                test_results['basic_ops'] = True
            except Exception as e:
                print(f"失败: {e}")
                test_results['basic_ops'] = False

            # 测试3: 矩阵乘法（带超时）
            print("矩阵乘法运算", end=' ... ')
            success = self._test_matmul_with_timeout()
            print("成功" if success else "失败或超时")
            test_results['matmul'] = success

            print_subheader("性能基准测试")

            # 性能测试
            self._run_performance_benchmark()

            print_subheader("显存管理测试")

            # 显存测试
            self._test_memory_management()

            torch.cuda.empty_cache()

            # 汇总结果
            passed_tests = sum(test_results.values())
            total_tests = len(test_results)

            print(f"\nPyTorch测试汇总: {passed_tests}/{total_tests} 项通过")
            return passed_tests >= total_tests * 0.6  # 60%通过率

        except Exception as e:
            print(f"PyTorch兼容性测试失败: {e}")
            return False

    def _test_matmul_with_timeout(self, timeout=10):
        """带超时的矩阵乘法测试"""
        result_queue = queue.Queue()

        def matmul_worker():
            try:
                import torch
                # 自适应测试大小
                for size in [64, 128, 256, 512]:
                    try:
                        a = torch.randn(size, size, device='cuda')
                        b = torch.randn(size, size, device='cuda')
                        c = torch.mm(a, b)
                        torch.cuda.synchronize()
                        result_queue.put((True, size))
                        return
                    except:
                        continue
                result_queue.put((False, 0))
            except Exception as e:
                result_queue.put((False, str(e)))

        thread = threading.Thread(target=matmul_worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)

        try:
            return result_queue.get_nowait()[0]
        except queue.Empty:
            return False

    def _run_performance_benchmark(self):
        """运行性能基准测试"""
        try:
            import torch
            import time

            # 找到合适的测试大小
            test_size = 256
            for size in [128, 256, 512, 1024, 2048, 4096, 8192]:
                try:
                    test_tensor = torch.randn(size, size, device='cuda')
                    torch.cuda.synchronize()
                    del test_tensor
                    test_size = size
                except:
                    break

            print(f"使用测试矩阵大小: {test_size}x{test_size}")

            # CPU基准
            cpu_tensor = torch.randn(test_size, test_size)
            start_time = time.time()
            cpu_result = torch.mm(cpu_tensor, cpu_tensor)
            cpu_time = time.time() - start_time

            # GPU基准
            gpu_tensor = torch.randn(test_size, test_size, device='cuda')
            torch.cuda.synchronize()
            start_time = time.time()
            gpu_result = torch.mm(gpu_tensor, gpu_tensor)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time

            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"CPU时间: {cpu_time*1000:.2f}ms")
                print(f"GPU时间: {gpu_time*1000:.2f}ms")
                print(f"加速比: {speedup:.2f}x")
            else:
                print("GPU运算极快，无法准确测量")

        except Exception as e:
            print(f"性能测试失败: {e}")

    def _test_memory_management(self):
        """测试显存管理"""
        try:
            import torch

            # 显存状态
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                total = torch.cuda.get_device_properties(0).total_memory / 1024**2

                print(f"显存使用情况:")
                print(f"  已分配: {allocated:.1f} MB")
                print(f"  已保留: {reserved:.1f} MB")
                print(f"  总显存: {total:.1f} MB")
                print(f"  使用率: {(allocated/total)*100:.1f}%")

                # 测试显存分配和释放
                print("显存分配测试", end=' ... ')
                tensors = []
                try:
                    for i in range(10):
                        tensor = torch.randn(100, 100, device='cuda')
                        tensors.append(tensor)

                    # 清理
                    del tensors
                    torch.cuda.empty_cache()
                    print("成功")
                except Exception as e:
                    print(f"失败: {e}")

        except Exception as e:
            print(f"显存测试失败: {e}")

    def check_available_versions(self):
        """检查可用的PyTorch、TensorFlow和CUDA版本"""
        print_header("可用版本检查")

        print_subheader("PyTorch可用版本")

        # 检查conda是否可用
        result, error = self._run_conda_command(["search", "pytorch", "-c", "pytorch"], timeout=30)

        if result and result.returncode == 0:
            versions = set()
            for line in result.stdout.split('\n'):
                if 'pytorch' in line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        version = parts[1]
                        if re.match(r'\d+\.\d+\.\d+', version):
                            versions.add(version)

            if versions:
                print("最近的PyTorch版本:")
                for version in sorted(versions, reverse=True)[:10]:
                    print(f"  - pytorch=={version}")
            else:
                print("未找到PyTorch版本信息")
        else:
            print(f"无法检查PyTorch版本: {error if error else '命令执行失败'}")

        print_subheader("TensorFlow可用版本")

        # 检查TensorFlow版本
        result, error = self._run_conda_command(["search", "tensorflow", "-c", "conda-forge"], timeout=30)

        if result and result.returncode == 0:
            tf_versions = set()
            for line in result.stdout.split('\n'):
                if 'tensorflow' in line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        version = parts[1]
                        if re.match(r'\d+\.\d+\.\d+', version):
                            tf_versions.add(version)

            if tf_versions:
                print("最近的TensorFlow版本:")
                for version in sorted(tf_versions, reverse=True)[:10]:
                    print(f"  - tensorflow=={version}")
        else:
            print("TensorFlow版本检查可通过pip获得:")
            print("  - pip install tensorflow (最新CPU版)")
            print("  - pip install tensorflow-gpu (如果还可用)")

        print_subheader("CUDA版本支持")

        # 检查pytorch-cuda包
        result, error = self._run_conda_command(["search", "pytorch-cuda", "-c", "pytorch", "-c", "nvidia"], timeout=30)

        if result and result.returncode == 0:
            cuda_versions = set()
            for line in result.stdout.split('\n'):
                if 'pytorch-cuda' in line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        version = parts[1]
                        if re.match(r'\d+\.\d+', version):
                            cuda_versions.add(version)

            if cuda_versions:
                print("可用的CUDA版本:")
                for version in sorted(cuda_versions, reverse=True):
                    print(f"  - pytorch-cuda={version}")

        # TensorFlow CUDA兼容性说明
        print("\nTensorFlow CUDA兼容性:")
        print("  - TensorFlow 2.15+: 支持CUDA 12.x")
        print("  - TensorFlow 2.10-2.14: 支持CUDA 11.x")
        print("  - TensorFlow 2.6-2.9: 支持CUDA 11.x")

    def check_conda_environment(self):
        """检查conda环境"""
        print_header("Conda环境检查")

        # 当前环境信息
        print(f"Python环境路径: {sys.prefix}")

        # 检查conda相关环境变量
        conda_prefix = os.environ.get('CONDA_PREFIX')
        conda_default_env = os.environ.get('CONDA_DEFAULT_ENV')

        if conda_prefix:
            print(f"CONDA_PREFIX: {conda_prefix}")
        if conda_default_env:
            print(f"当前环境名称: {conda_default_env}")

        # 尝试找到conda可执行文件
        conda_exe = self._find_conda_executable()
        if conda_exe:
            print(f"Conda可执行文件路径: {conda_exe}")

            # 获取环境列表
            result, error = self._run_conda_command(["env", "list"])
            if result and result.returncode == 0:
                print("\nConda环境列表:")
                for line in result.stdout.split('\n'):
                    if line.strip() and not line.startswith('#'):
                        if '*' in line:
                            print(f"  {line.strip()} <- 当前环境")
                        else:
                            print(f"  {line.strip()}")

            # 检查已安装的相关包
            result, error = self._run_conda_command(["list"])
            if result and result.returncode == 0:
                relevant_packages = []
                for line in result.stdout.split('\n'):
                    if any(pkg in line.lower() for pkg in ['torch', 'tensorflow', 'cuda', 'nvidia']):
                        if not line.startswith('#') and line.strip():
                            relevant_packages.append(line.strip())

                if relevant_packages:
                    print("\n已安装的相关包:")
                    for pkg in relevant_packages:
                        print(f"  {pkg}")
        else:
            print("未找到conda可执行文件")
            print("可能的原因:")
            print("  - conda未正确安装")
            print("  - conda不在系统PATH中")
            print("  - 使用的是pip虚拟环境而非conda环境")

            # 检查pip包作为备选
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "list"],
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    relevant_packages = []
                    for line in result.stdout.split('\n'):
                        if any(pkg in line.lower() for pkg in ['torch', 'tensorflow', 'cuda', 'nvidia']):
                            relevant_packages.append(line.strip())

                    if relevant_packages:
                        print("\n通过pip检查到的相关包:")
                        for pkg in relevant_packages:
                            print(f"  {pkg}")
            except Exception as e:
                print(f"pip包检查也失败: {e}")

    def generate_recommendations(self):
        """生成个性化推荐"""
        print_header("个性化推荐方案")

        has_driver = bool(self.driver_info.get('version'))
        has_pytorch = self.pytorch_info.get('installed', False)
        has_pytorch_cuda = self.pytorch_info.get('cuda_available', False)
        has_tensorflow = self.tensorflow_info.get('installed', False)
        has_tensorflow_gpu = self.tensorflow_info.get('gpu_available', False)

        if not has_driver:
            print("❌ 问题: 未检测到NVIDIA驱动")
            print("解决方案:")
            print("1. 从NVIDIA官网下载并安装最新驱动")
            print("2. 重启计算机")
            print("3. 重新运行此检测工具")
            print("注意: 在虚拟化环境中(如Docker容器)，这可能是正常现象")
            return

        # PyTorch检查
        if not has_pytorch:
            print("❌ 问题: PyTorch未安装")
            self._suggest_pytorch_installation()
        elif not has_pytorch_cuda:
            print("❌ 问题: PyTorch不支持CUDA")
            print("当前PyTorch信息:")
            print(f"  版本: {self.pytorch_info.get('version', '未知')}")
            print(f"  CUDA编译版本: {self.pytorch_info.get('cuda_compiled', '无')}")
            print()
            self._suggest_pytorch_reinstallation()

        # TensorFlow检查
        print_subheader("TensorFlow状态")
        if not has_tensorflow:
            print("❌ 问题: TensorFlow未安装")
            self._suggest_tensorflow_installation()
        elif not has_tensorflow_gpu:
            print("❌ 问题: TensorFlow不支持GPU")
            print("当前TensorFlow信息:")
            print(f"  版本: {self.tensorflow_info.get('version', '未知')}")
            print(f"  CUDA编译版本: {self.tensorflow_info.get('cuda_compiled', '无')}")
            print()
            self._suggest_tensorflow_gpu_setup()
        else:
            print("✅ TensorFlow GPU支持正常")

        # 检查兼容性问题
        if has_pytorch_cuda and has_tensorflow_gpu:
            self._check_framework_compatibility()

        print_subheader("环境优化建议")
        if has_pytorch_cuda or has_tensorflow_gpu:
            print("✅ 基础环境正常")
            self._suggest_optimizations()

    def _suggest_tensorflow_installation(self):
        """建议TensorFlow安装方案"""
        max_cuda = self.driver_info.get('max_cuda', '11.8')

        print("TensorFlow安装推荐:")
        print()

        if max_cuda.startswith('12.'):
            print("方案1: 最新TensorFlow + CUDA 12.x")
            print("pip install tensorflow")
            print()
            print("方案2: 指定版本 (更稳定)")
            print("pip install tensorflow==2.15.0")
        else:
            print("方案1: TensorFlow 2.10-2.14 (CUDA 11.x)")
            print("pip install tensorflow==2.10.0")
            print()
            print("方案2: 最新版本 (自动适配)")
            print("pip install tensorflow")

        conda_exe = self._find_conda_executable()
        if conda_exe:
            print()
            print("使用conda安装:")
            print("conda install tensorflow -c conda-forge")

    def _suggest_tensorflow_gpu_setup(self):
        """建议TensorFlow GPU设置"""
        print("TensorFlow GPU配置建议:")
        print()
        print("1. 检查CUDA和cuDNN兼容性:")
        tf_version = self.tensorflow_info.get('version', '')
        print(f"   当前TensorFlow版本: {tf_version}")

        if tf_version:
            major_version = tf_version.split('.')[0]
            minor_version = tf_version.split('.')[1] if '.' in tf_version else '0'

            if major_version == '2' and int(minor_version) >= 15:
                print("   推荐CUDA版本: 12.x")
                print("   推荐cuDNN版本: 8.8+")
            elif major_version == '2' and int(minor_version) >= 10:
                print("   推荐CUDA版本: 11.8")
                print("   推荐cuDNN版本: 8.6+")
            else:
                print("   建议升级到TensorFlow 2.10+")

        print()
        print("2. 验证GPU可用性:")
        print("   python -c \"import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))\"")
        print()
        print("3. 重新安装TensorFlow GPU版本:")
        print("   pip uninstall tensorflow")
        print("   pip install tensorflow==2.11.0")

    def _check_framework_compatibility(self):
        """检查框架间兼容性"""
        print_subheader("框架兼容性检查")

        pytorch_cuda = self.pytorch_info.get('cuda_compiled', '')
        tensorflow_cuda = self.tensorflow_info.get('cuda_compiled', '')

        print(f"PyTorch CUDA版本: {pytorch_cuda}")
        print(f"TensorFlow CUDA版本: {tensorflow_cuda}")

        if pytorch_cuda and tensorflow_cuda:
            pytorch_major = pytorch_cuda.split('.')[0]
            tensorflow_major = tensorflow_cuda.split('.')[0]

            if pytorch_major == tensorflow_major:
                print("✅ CUDA版本兼容")
            else:
                print("⚠️ CUDA版本差异可能导致冲突")
                print("建议:")
                print("  - 统一使用相同的CUDA主版本")
                print("  - 在不同项目中分别使用不同的conda环境")

    def _suggest_pytorch_installation(self):
        """建议PyTorch安装方案"""
        max_cuda = self.driver_info.get('max_cuda', '11.8')

        print("PyTorch安装推荐:")
        print()

        conda_exe = self._find_conda_executable()
        if conda_exe:
            print("使用conda安装:")
            if max_cuda.startswith('12.'):
                print("方案1: 最新版本 + CUDA 12.1")
                print("conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
                print()
                print("方案2: 稳定版本 + CUDA 11.8")
                print("conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
            else:
                print("方案1: 稳定版本 + CUDA 11.8")
                print("conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
        else:
            print("使用pip安装:")
            if max_cuda.startswith('12.'):
                print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            else:
                print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

    def _suggest_pytorch_reinstallation(self):
        """建议PyTorch重新安装"""
        max_cuda = self.driver_info.get('max_cuda', '11.8')

        print("PyTorch重新安装步骤:")
        print()

        conda_exe = self._find_conda_executable()
        if conda_exe:
            print("使用conda重新安装:")
            print("1. 卸载当前版本:")
            print("   conda uninstall pytorch torchvision torchaudio pytorch-cuda cudatoolkit")
            print()
            print("2. 安装CUDA版本:")
            if max_cuda.startswith('12.'):
                print("   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
            else:
                print("   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
        else:
            print("使用pip重新安装:")
            print("1. 卸载当前版本:")
            print("   pip uninstall torch torchvision torchaudio")
            print()
            print("2. 安装CUDA版本:")
            if max_cuda.startswith('12.'):
                print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            else:
                print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

    def _suggest_optimizations(self):
        """建议优化方案"""
        optimizations = []

        # 根据GPU显存建议
        for gpu_info in self.gpu_info.values():
            memory_gb = gpu_info.get('memory_total', 0) / 1024
            if memory_gb < 6:
                optimizations.append("显存较小，建议使用较小的batch size和模型")
            elif memory_gb > 20:
                optimizations.append("显存充足，可以使用大模型和混合精度训练")

        # 环境变量优化
        optimizations.extend([
            "设置环境变量优化GPU性能:",
            "  export CUDA_VISIBLE_DEVICES=0  # 指定使用的GPU",
            "  export TF_FORCE_GPU_ALLOW_GROWTH=true  # TensorFlow内存增长",
            "",
            "代码优化建议:",
            "  - 使用torch.cuda.amp进行混合精度训练 (PyTorch)",
            "  - 使用tf.config.experimental.set_memory_growth() (TensorFlow)",
            "  - 设置pin_memory=True在DataLoader中加速数据传输",
            "  - 使用torch.compile()加速模型推理 (PyTorch 2.0+)"
        ])

        for opt in optimizations:
            print(f"  {opt}")

    def run_comprehensive_test(self):
        """运行综合测试"""
        print_header("综合GPU测试")

        test_results = {
            'pytorch': False,
            'tensorflow': False
        }

        # PyTorch测试
        if self.pytorch_info.get('cuda_available', False):
            print_subheader("PyTorch GPU测试")
            test_results['pytorch'] = self._run_pytorch_tests()

        # TensorFlow测试
        if self.tensorflow_info.get('gpu_available', False):
            print_subheader("TensorFlow GPU测试")
            test_results['tensorflow'] = self.run_tensorflow_compatibility_test()

        # 框架间互操作测试
        if test_results['pytorch'] and test_results['tensorflow']:
            print_subheader("框架互操作测试")
            self._test_framework_interop()

        return test_results

    def _test_framework_interop(self):
        """测试框架间互操作"""
        try:
            import torch
            import tensorflow as tf
            import numpy as np

            print("测试PyTorch和TensorFlow数据转换", end=' ... ')

            # PyTorch张量
            torch_tensor = torch.randn(100, 100).cuda()

            # 转换为numpy
            numpy_array = torch_tensor.cpu().numpy()

            # 转换为TensorFlow
            with tf.device('/GPU:0'):
                tf_tensor = tf.constant(numpy_array)
                tf_result = tf.matmul(tf_tensor, tf_tensor)

            # 转换回PyTorch
            back_to_torch = torch.from_numpy(tf_result.numpy()).cuda()

            print("成功")
            print("✅ 框架间数据转换正常")

        except Exception as e:
            print(f"失败: {e}")
            print("⚠️ 框架间可能存在冲突")

    def export_report(self, filename="gpu_detection_report.txt"):
        """导出检测报告"""
        print_header("导出检测报告")

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("GPU环境检测报告\n")
                f.write("=" * 50 + "\n\n")

                # 系统信息
                f.write("系统信息:\n")
                f.write(f"  操作系统: {platform.system()} {platform.release()}\n")
                f.write(f"  Python版本: {sys.version.split()[0]}\n")
                f.write(f"  Python路径: {sys.executable}\n\n")

                # 驱动信息
                if self.driver_info:
                    f.write("NVIDIA驱动信息:\n")
                    f.write(f"  驱动版本: {self.driver_info.get('version', 'N/A')}\n")
                    f.write(f"  支持CUDA版本: {self.driver_info.get('max_cuda', 'N/A')}\n\n")

                # GPU信息
                if self.gpu_info:
                    f.write("GPU信息:\n")
                    for gpu_id, gpu_data in self.gpu_info.items():
                        f.write(f"  {gpu_id}: {gpu_data.get('name', 'Unknown')}\n")
                        f.write(f"    显存: {gpu_data.get('memory_total', 0)/1024:.1f}GB\n")

                # PyTorch信息
                if self.pytorch_info.get('installed'):
                    f.write("PyTorch信息:\n")
                    f.write(f"  版本: {self.pytorch_info.get('version', 'N/A')}\n")
                    f.write(f"  CUDA支持: {self.pytorch_info.get('cuda_available', False)}\n")
                    f.write(f"  CUDA版本: {self.pytorch_info.get('cuda_compiled', 'N/A')}\n\n")

                # TensorFlow信息
                if self.tensorflow_info.get('installed'):
                    f.write("TensorFlow信息:\n")
                    f.write(f"  版本: {self.tensorflow_info.get('version', 'N/A')}\n")
                    f.write(f"  GPU支持: {self.tensorflow_info.get('gpu_available', False)}\n")
                    f.write(f"  CUDA版本: {self.tensorflow_info.get('cuda_compiled', 'N/A')}\n")
                    f.write(f"  cuDNN版本: {self.tensorflow_info.get('cudnn_compiled', 'N/A')}\n")

            print(f"✅ 报告已导出到: {filename}")

        except Exception as e:
            print(f"❌ 导出报告失败: {e}")

    def run_full_detection(self):
        """运行完整检测流程"""
        print("=" * 60)
        print("CUDA环境全面检测工具 v2.0")
        print("支持PyTorch + TensorFlow双框架检测")
        print("=" * 60)

        # 执行所有检测步骤
        self.detect_system_info()
        has_driver = self.detect_nvidia_driver()
        self.detect_cuda_runtime()
        has_pytorch_cuda = self.detect_pytorch_env()
        has_tensorflow_gpu = self.detect_tensorflow_env()

        # 运行兼容性测试
        if has_driver and (has_pytorch_cuda or has_tensorflow_gpu):
            self.run_comprehensive_test()

        # 检查可用版本和环境
        self.check_available_versions()
        self.check_conda_environment()

        # 生成建议
        self.generate_recommendations()

        # 导出报告
        self.export_report()

        print_header("检测完成")
        print("如需更详细的帮助，请提供具体的错误信息或需求")

def main():
    detector = CUDADetector()
    detector.run_full_detection()

if __name__ == "__main__":
    main()
