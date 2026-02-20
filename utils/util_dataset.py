import os
import logging
import inspect
from typing import Dict

logger = logging.getLogger("sys_log")

def parse_scps(scp_path: str) -> Dict[str, str]:
    """
    Robustly parses .scp manifest files into a dictionary mapping keys to file paths.
    """
    scp_dict = {}
    func_name = inspect.currentframe().f_code.co_name
    
    try:
        if not os.path.exists(scp_path):
            raise FileNotFoundError(f"Manifest file not found: {scp_path}")
            
        with open(scp_path, 'r', encoding='utf-8') as f:
            for line_idx, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line:
                    continue
                
                tokens = line.split(maxsplit=1)
                
                if len(tokens) != 2:
                    error_msg = f"Malformed SCP entry at {scp_path}:{line_idx} -> '{line}'"
                    raise RuntimeError(error_msg)
                
                key, addr = tokens
                
                if key in scp_dict:
                    raise ValueError(f"Duplicate key '{key}' detected in manifest: {scp_path}")
                
                scp_dict[key] = addr
                
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logger.error(f"Error in {func_name}: {e}")
        raise
    except Exception as e:
        logger.critical(f"Unexpected System Error during SCP parsing: {e}")
        raise
    finally:
        logger.debug(f"Execution finished: {__name__}.{func_name}")
        
    return scp_dict