#!/usr/bin/env python3
"""LoRA management and switching utility."""

import argparse
from pathlib import Path

import yaml


class LoRAManager:
    """Manage multiple LoRA models."""

    def __init__(self, config_path: Path = Path("config/config.yaml")):
        self.config_path = config_path
        self.lora_dir = Path("models/lora")

    def list_loras(self):
        """List all available LoRAs."""
        print("🎨 Available LoRA Models:")
        print("=" * 60)

        if not self.lora_dir.exists():
            print("No LoRA models found.")
            return

        loras = [
            d
            for d in self.lora_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        if not loras:
            print("No LoRA models found.")
            return

        for idx, lora_path in enumerate(sorted(loras), 1):
            print(f"{idx}. {lora_path.name}")

            # Check for adapter files
            has_model = any(lora_path.glob("adapter_*.safetensors")) or any(
                lora_path.glob("*.safetensors")
            )
            status = "✅ Ready" if has_model else "⚠️  Empty/Training"
            print(f"   Status: {status}")
            print(f"   Path: {lora_path}")
            print()

    def get_current_lora(self):
        """Get currently active LoRA from config."""
        if not self.config_path.exists():
            return None

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        lora_path = config.get("model", {}).get("lora_path")
        lora_scale = config.get("model", {}).get("lora_scale", 0.8)

        return lora_path, lora_scale

    def set_lora(self, lora_name: str = None, scale: float = 0.8):
        """Set active LoRA in config."""
        if not self.config_path.exists():
            print(f"❌ Config file not found: {self.config_path}")
            return False

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        if lora_name is None:
            # Disable LoRA
            config["model"]["lora_path"] = None
            config["model"]["lora_scale"] = 0.8
            print("✅ LoRA disabled - using base model only")
        else:
            # Enable LoRA
            lora_path = f"models/lora/{lora_name}"
            if not Path(lora_path).exists():
                print(f"❌ LoRA not found: {lora_path}")
                return False

            config["model"]["lora_path"] = lora_path
            config["model"]["lora_scale"] = scale
            print(f"✅ LoRA set to: {lora_name}")
            print(f"   Scale: {scale}")

        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return True

    def show_status(self):
        """Show current LoRA status."""
        result = self.get_current_lora()

        print("🎨 Current LoRA Status:")
        print("=" * 60)

        if result is None or result[0] is None:
            print("Status: ❌ No LoRA active (using base model)")
        else:
            lora_path, scale = result
            lora_name = Path(lora_path).name
            print("Status: ✅ LoRA active")
            print(f"Model: {lora_name}")
            print(f"Scale: {scale}")
            print(f"Path: {lora_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Manage LoRA models")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    subparsers.add_parser("list", help="List all available LoRAs")

    # Status command
    subparsers.add_parser("status", help="Show current LoRA status")

    # Set command
    set_parser = subparsers.add_parser("set", help="Set active LoRA")
    set_parser.add_argument("name", help="LoRA name (or 'none' to disable)")
    set_parser.add_argument(
        "--scale", type=float, default=0.8, help="LoRA scale (0.0-1.0)"
    )

    args = parser.parse_args()

    manager = LoRAManager()

    if args.command == "list":
        manager.list_loras()
    elif args.command == "status":
        manager.show_status()
    elif args.command == "set":
        if args.name.lower() == "none":
            manager.set_lora(None)
        else:
            manager.set_lora(args.name, args.scale)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
