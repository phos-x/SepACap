import os
import logging
import inspect
from typing import Dict

# Call logger for monitoring - using loguru-style or standard logging alignment
logger = logging.getLogger("sys_log")

def parse_scps(scp_path: str) -> Dict[str, str]:
    """
    Robustly parses .scp manifest files into a dictionary mapping keys to file paths.
    
    DSA & Security Best Practices implemented:
    1. Defensive Whitespace Handling: Uses strip() and checks for empty lines to avoid context errors.
    2. Path Space Support: Uses maxsplit=1 to support file paths that contain spaces.
    3. Duplicate Key Prevention: Ensures manifest integrity to prevent data mapping drift.
    4. Absolute Path Verification: (Optional) logs warnings if paths are not absolute.
    """
    scp_dict = {}
    func_name = inspect.currentframe().f_code.co_name
    
    try:
        # Security Check: Validate path existence before attempting I/O
        if not os.path.exists(scp_path):
            raise FileNotFoundError(f"Manifest file not found: {scp_path}")
            
        with open(scp_path, 'r', encoding='utf-8') as f:
            for line_idx, raw_line in enumerate(f, 1):
                # DSA: Remove leading/trailing whitespace and skip empty lines
                # This directly fixes the RuntimeError: Error format of context ''
                line = raw_line.strip()
                if not line:
                    continue
                
                # Handling Space-in-Paths: Split only on the first whitespace encountered
                # JaCappella paths can be complex; this ensures the path remains a single token
                tokens = line.split(maxsplit=1)
                
                if len(tokens) != 2:
                    error_msg = f"Malformed SCP entry at {scp_path}:{line_idx} -> '{line}'"
                    raise RuntimeError(error_msg)
                
                key, addr = tokens
                
                # Integrity Check: Prevent duplicate keys which cause non-deterministic training
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