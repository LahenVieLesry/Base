import logging
import os
import shutil

# 默认日志目录
LOG_DIR = 'logs'

# 创建日志目录（如果不存在）
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def copy_config_to_log_file(log_file_path, config_file_path):
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as config_file:
            config_content = config_file.read()
        with open(log_file_path, 'w') as log_file:
            log_file.write(config_content)
            log_file.write('\n\n')  # 添加两个回车

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    log_file_path = os.path.join(LOG_DIR, f'{name}.log')
    # 如果日志文件存在，先复制配置文件内容到日志文件开头
    if not os.path.exists(log_file_path):
        copy_config_to_log_file(log_file_path, 'config/config.yaml')

    # 文件处理器
    file_handler = logging.FileHandler(log_file_path, 'a')
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 清除现有的处理器
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log(name, msg):
    """记录日志信息到指定文件和控制台"""
    logger = get_logger(name)
    logger.info(msg)