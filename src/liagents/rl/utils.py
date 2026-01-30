def check_trl_installation() -> bool:
    """
    检查TRL是否已安装
    
    Returns:
        是否已安装TRL
    """
    try:
        import trl
        return True
    except ImportError:
        return False