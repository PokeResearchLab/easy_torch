import torch
from typing import List, Tuple, Dict

# Class added because torch.nn.ModuleDict doesn't allow certain keys to be used if they conflict with existing class attributes
# https://github.com/pytorch/pytorch/issues/71203
SUFFIX = "____"
SUFFIX_LENGTH = len(SUFFIX)
class RobustModuleDict(torch.nn.Module):
    """Torch ModuleDict wrapper that permits keys with any name.

    Torch's ModuleDict doesn't allow certain keys to be used if they
    conflict with existing class attributes, e.g.

    > torch.nn.ModuleDict({'type': torch.nn.Module()})  # Raises KeyError.

    This class is a simple wrapper around torch's ModuleDict that
    mitigates possible conflicts by using a key-suffixing protocol.
    """

    def __init__(self, init_dict: Dict[str, torch.nn.Module] = None) -> None:
        super().__init__()
        self.module_dict = torch.nn.ModuleDict()
        if init_dict is not None:
            self.update(init_dict)

    def __getitem__(self, key) -> torch.nn.Module:
        return self.module_dict[key + SUFFIX]

    def __setitem__(self, key: str, module: torch.nn.Module) -> None:
        self.module_dict[key + SUFFIX] = module

    def __len__(self) -> int:
        return len(self.module_dict)

    def keys(self) -> List[str]:
        return [key[:-SUFFIX_LENGTH] for key in self.module_dict.keys()]

    def values(self) -> List[torch.nn.Module]:
        return [module for _, module in self.module_dict.items()]

    def items(self) -> List[Tuple[str, torch.nn.Module]]:
        return [
            (key[:-SUFFIX_LENGTH], 
             module) for key, module in self.module_dict.items()
        ]

    def update(self, modules: Dict[str, torch.nn.Module]) -> None:
        for key, module in modules.items():
            self[key] = module

    def __next__(self) -> None:
        return next(iter(self))

    def __iter__(self) -> None:
        return iter(self.keys())
    

import time
import logging, subprocess, platform


# Create logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set logger to capture INFO level and above

# --- GPU Temperature Check Function ---
def get_nvidia_gpu_temp():
    """
    Gets the current temperature of the NVIDIA GPU(s).

    Returns:
        int: The maximum temperature found among all NVIDIA GPUs in Celsius,
             or None if nvidia-smi is not found or fails.
    """
    if platform.system() == "Windows":
        # Try the default install path for nvidia-smi on Windows
        nvidia_smi_path = r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
        command = f'"{nvidia_smi_path}" --query-gpu=temperature.gpu --format=csv,noheader,nounits'
    elif platform.system() == "Linux":
        command = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"

    try:
        # Execute the command, capture output, decode to text
        # Use timeout to prevent hanging if nvidia-smi gets stuck
        output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.PIPE, timeout=15)

        # Split lines in case of multiple GPUs, strip whitespace, convert to int
        temps = [int(t.strip()) for t in output.strip().split('\n') if t.strip().isdigit()] # Ensure it's a digit

        if not temps:
            logger.warning("GPU Temp Check: nvidia-smi command executed but returned no parsable temperature.")
            return None

        # Return the maximum temperature if multiple GPUs are present
        return max(temps)

    except FileNotFoundError:
        logger.error("GPU Temp Check: 'nvidia-smi' command not found.")
        logger.error("Ensure NVIDIA drivers are installed and 'nvidia-smi' is in your system's PATH")
        if platform.system() == "Windows":
            logger.error(f" (Expected default path: {nvidia_smi_path})")
        return None
    except subprocess.TimeoutExpired:
         logger.warning("GPU Temp Check: 'nvidia-smi' command timed out.")
         return None
    except subprocess.CalledProcessError as e:
        logger.error(f"GPU Temp Check: Error executing nvidia-smi: {e}", exc_info=False)
        if e.stderr: logger.error(f"stderr: {e.stderr.strip()}")
        return None
    except ValueError as e:
        logger.error(f"GPU Temp Check: Could not parse temperature from nvidia-smi output. Error: {e}", exc_info=False)
        logger.error(f"nvidia-smi output was: '{output.strip()}'")
        return None
    except Exception as e: # Catch unexpected errors
        logger.error(f"GPU Temp Check: An unexpected error occurred: {e}", exc_info=True)
        return None
# --- End GPU Temp Function ---


MAX_GPU_TEMP_THRESHOLD = 88
COOL_DOWN_WAIT_SECONDS = 2*60 

# --- <<< GPU Temperature Check Loop (Moved Here) >>> ---
logger.info(f"--- Checking GPU temperature before starting temp {temp} ---") # File only
while True:
    current_temp = get_nvidia_gpu_temp()
    if current_temp is None:
        logger.warning(f"Could not determine GPU temperature before temp {temp}. Proceeding with caution.") # Console & File
        break

    logger.info(f"Current max GPU Temp: {current_temp}°C (Threshold: {MAX_GPU_TEMP_THRESHOLD}°C)") # File only
    if current_temp < MAX_GPU_TEMP_THRESHOLD:
        logger.info(f"GPU temperature OK. Proceeding.") # File only
        break # Temperature is fine, exit check loop
    else:
        logger.warning(f"GPU temp ({current_temp}°C) >= threshold ({MAX_GPU_TEMP_THRESHOLD}°C).") # Console & File
        logger.warning(f"Waiting for {COOL_DOWN_WAIT_SECONDS} seconds...") # Console & File
        try:
            time.sleep(COOL_DOWN_WAIT_SECONDS)
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt during cool down wait. Re-raising.") # Console & File
            raise # Re-raise to be caught by main handler & exit script
        logger.info("Re-checking temperature...") # File only
        # Loop continues to check again
# --- <<< End GPU Temperature Check Loop >>> ---

logger.info(f"-- Processing Temperature: {temp} --")