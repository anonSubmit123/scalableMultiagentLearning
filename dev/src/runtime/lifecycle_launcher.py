import logging
import sys
import os
import json
import shlex
from typing import Any, Dict
import paramiko
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)

def launch_lifecylce_servers(lifecycle_config: Dict[str, Any], ssh_user: str = "admin",
    host_basepath: str = "~/util/project") -> None:
    physical_systems = lifecycle_config.get("physical_systems")
    if not physical_systems or not isinstance(physical_systems, list):
        logger.error("Invalid configuration layout. Missing 'physical_systems' array.")
        return

    lifecycle_module = "runtime.host_lifecycle_server"
    lifecycle_launcher_path = f"{host_basepath}/util/lifecycle_launcher.sh"
    for system in physical_systems:
        try:
            target_host = system["host"]
            worker_id = system["id"]
            target_port = system["lifecycle_endpoint"]["port"]
            lifecycle_host = system["lifecycle_endpoint"].get("host", target_host)
        except KeyError as ke:
            logger.error(f"Skipping entry. Missing required field: {ke}")
            continue

        logger.info(f"Processing system '{worker_id}' at {target_host}:{target_port}")

        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            ssh.connect(hostname=target_host, username=ssh_user, timeout=10)

            launch_args = [
                lifecycle_launcher_path,
                "--project-dir", host_basepath,
                "--module", lifecycle_module,
                "--physical-system-id", worker_id,
                "--host", lifecycle_host,
                "--port", str(target_port),
            ]
            launch_cmd = " ".join(shlex.quote(arg) for arg in launch_args)
            _, stdout, stderr = ssh.exec_command(launch_cmd)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                logger.error(f"-> Lifecycle launch failed: {stderr.read().decode().strip()}")
            else:
                logger.info(f"-> {stdout.read().decode().strip()}")

            ssh.close()

        except paramiko.AuthenticationException:
            logger.error(f"SSH Authentication failed for {ssh_user}@{target_host}.")
        except paramiko.SSHException as ssh_err:
            logger.error(f"SSH network connection layer failure on {target_host}: {ssh_err}")
        except Exception as err:
            logger.error(f"Unexpected runtime error managing {target_host}: {err}")


def configure_logging(log_name: str) -> None:
      log_dir = Path("logs")
      log_dir.mkdir(exist_ok=True)

      logging.basicConfig(
          level=logging.INFO,
          format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s",
          handlers=[
              logging.StreamHandler(sys.stdout),
              logging.FileHandler(log_dir / f"{log_name}.log", mode="a", encoding="utf-8"),
          ],
          force=True,
      )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch lifecycle servers.")
    parser.add_argument("--user", type=str, required=False, default="admin")
    parser.add_argument("--basepath", type=str, required=False, default="~/util/project")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = args.config
    configure_logging("lifecycle_launcher")

    if not config_path or not os.path.exists(config_path):
        logger.error(f"Configuration file not found at path: '{config_path}'")
        sys.exit(1)

    config_data = {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

    except json.JSONDecodeError as jde:
        logger.error(
            f"Syntax error inside JSON file! Check for missing commas or quotes.\n"
            f"Details: {jde.msg} at line {jde.lineno}, column {jde.colno}"
        )
        sys.exit(1)

    except PermissionError:
        logger.error(f"Permission denied! Current user lacks read access to: '{config_path}'")
        sys.exit(1)

    except Exception as err:
        logger.error(f"Unexpected error while opening configuration file: {err}")
        sys.exit(1)

    logger.info("Executing server orchestration block...")
    launch_lifecylce_servers(lifecycle_config = config_data, ssh_user = args.user,
        host_basepath = args.basepath)
    logger.info("Orchestration pipeline execution complete.")
