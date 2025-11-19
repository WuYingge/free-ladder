import logging
import logging.handlers
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import inspect
import threading

class CustomLogger:
    """
    自定义日志记录器，支持多进程多线程，自动创建日志文件，过滤堆栈信息
    """
    
    def __init__(self, 
                 log_dir: str = "logs", 
                 log_name: Optional[str] = None,
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 package_filter: Optional[str] = None):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志存储目录
            log_name: 日志文件名（不含扩展名），如果为None则使用程序名
            max_bytes: 单个日志文件最大大小（字节）
            backup_count: 保留的备份文件数量
            package_filter: 只显示指定包名的堆栈信息，如果为None则显示所有
        """
        self.log_dir = Path(log_dir)
        self.package_filter = package_filter
        self._lock = threading.RLock()
        
        # 确保日志目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 确定日志文件名
        if log_name is None:
            log_name = Path(sys.argv[0]).stem if sys.argv else "app"
        
        # 创建带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{log_name}_{timestamp}.log"
        
        # 配置根日志记录器
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        
        # 移除所有现有处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(threadName)s]'
        )
        
        # 文件处理器 - 使用旋转文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 记录初始化信息
        self.info(f"日志系统初始化完成，日志文件: {self.log_file}")
    
    def _filter_stack(self, exc_info) -> str:
        """过滤堆栈跟踪，只显示指定包的堆栈信息"""
        if exc_info is None:
            return ""
            
        # 获取完整的堆栈跟踪
        stack_lines = traceback.format_exception(*exc_info)
        
        # 如果没有包过滤器，返回完整堆栈
        if not self.package_filter:
            return "".join(stack_lines)
        
        # 过滤堆栈，只保留包含包名的行
        filtered_stack = []
        for line in stack_lines:
            if self.package_filter in line:
                filtered_stack.append(line)
        
        # 如果过滤后没有堆栈信息，返回原始堆栈的前几行
        if not filtered_stack:
            return "".join(stack_lines[:3]) + "... [堆栈已过滤]\n"
        
        return "".join(filtered_stack)
    
    def _log_with_stack(self, level, msg, *args, exc_info=None, **kwargs):
        """带堆栈过滤的日志记录方法"""
        with self._lock:
            # 获取当前堆栈信息
            if exc_info:
                stack_trace = self._filter_stack(exc_info)
                msg = f"{msg}\n{stack_trace}"
            
            # 记录日志
            self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        """记录调试信息"""
        self._log_with_stack(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        """记录一般信息"""
        self._log_with_stack(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """记录警告信息"""
        self._log_with_stack(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """记录错误信息"""
        # 自动捕获异常信息
        exc_info = sys.exc_info()
        if any(exc_info):
            self._log_with_stack(logging.ERROR, msg, *args, exc_info=exc_info, **kwargs)
        else:
            self._log_with_stack(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        """记录严重错误信息"""
        # 自动捕获异常信息
        exc_info = sys.exc_info()
        if any(exc_info):
            self._log_with_stack(logging.CRITICAL, msg, *args, exc_info=exc_info, **kwargs)
        else:
            self._log_with_stack(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg, *args, **kwargs):
        """记录异常信息（自动包含堆栈跟踪）"""
        self._log_with_stack(logging.ERROR, msg, *args, exc_info=sys.exc_info(), **kwargs)
    
    def get_log_file(self) -> Path:
        """获取当前日志文件路径"""
        return self.log_file
    
    def list_log_files(self) -> list:
        """列出所有日志文件"""
        return sorted(self.log_dir.glob("*.log"))


# 全局日志记录器实例
_logger_instance = None

def setup_logging(log_dir: str = "logs", 
                 log_name: Optional[str] = None,
                 package_filter: Optional[str] = None) -> CustomLogger:
    """
    设置全局日志记录器
    
    Args:
        log_dir: 日志存储目录
        log_name: 日志文件名（不含扩展名）
        package_filter: 只显示指定包名的堆栈信息
    
    Returns:
        配置好的日志记录器实例
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = CustomLogger(
            log_dir=log_dir,
            log_name=log_name,
            package_filter=package_filter
        )
    return _logger_instance

def get_logger() -> CustomLogger:
    """
    获取全局日志记录器
    
    Returns:
        全局日志记录器实例
    """
    if _logger_instance is None:
        # 如果没有初始化，使用默认配置
        setup_logging()
    return _logger_instance
