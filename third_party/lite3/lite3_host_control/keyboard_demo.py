"""
绝影Lite3 上位机交互式键盘控制 Demo
使用 WASD 控制前后左右, QE 控制旋转, Space 急停
"""

import sys
import time
import threading

from lite3_controller import Lite3Controller, Twist, RobotStateUpload


HELP_TEXT = """
╔══════════════════════════════════════════════════════╗
║        绝影Lite3 键盘控制器                           ║
╠══════════════════════════════════════════════════════╣
║  W / S       前进 / 后退                              ║
║  A / D       左移 / 右移                              ║
║  Q / E       左转 / 右转                              ║
║  SPACE       急停!!                                   ║
║  U           起立/趴下                                ║
║  P           准备Twist控制(起立+自主+移动)              ║
║  1-3         低速/中速/高速步态                         ║
║  0           停止运动(零速度)                           ║
║  I           打印状态                                  ║
║  ESC / Ctrl+C  退出                                   ║
╚══════════════════════════════════════════════════════╝
"""

# 速度参数
VX_STEP = 0.2    # m/s
VY_STEP = 0.15   # m/s
WZ_STEP = 0.3    # rad/s


def main():
    print(HELP_TEXT)

    robot_ip = input("输入运动主机IP (默认 192.168.2.1): ").strip()
    if not robot_ip:
        robot_ip = "192.168.2.1"

    ctrl = Lite3Controller(robot_ip=robot_ip)

    # 注册状态回调
    def on_state(state: RobotStateUpload):
        pass  # 静默

    ctrl.register_state_callback(on_state)
    ctrl.start()

    # 当前速度
    vx, vy, wz = 0.0, 0.0, 0.0
    twist_active = False

    def send_current_twist():
        nonlocal twist_active
        if not twist_active:
            ctrl.start_twist_control(Twist(vx, vy, wz))
            twist_active = True
        else:
            ctrl.update_twist(Twist(vx, vy, wz))

    try:
        # 跨平台键盘读取
        if sys.platform == 'win32':
            import msvcrt
            def get_key():
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch == b'\xe0' or ch == b'\x00':
                        msvcrt.getch()
                        return None
                    return ch.decode('utf-8', errors='ignore').lower()
                return None
        else:
            import tty, termios, select
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            def get_key():
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    return sys.stdin.read(1).lower()
                return None

        print("\n等待按键输入... (按 P 一键准备)")

        while True:
            key = get_key()
            if key is None:
                time.sleep(0.05)
                continue

            if key == '\x1b' or key == '\x03':  # ESC or Ctrl+C
                break

            elif key == ' ':
                vx, vy, wz = 0.0, 0.0, 0.0
                ctrl.soft_estop()
                twist_active = False

            elif key == 'u':
                ctrl.stand_up_or_down()

            elif key == 'p':
                ctrl.prepare_for_twist_control()

            elif key == 'w':
                vx = min(1.0, vx + VX_STEP)
                send_current_twist()
                print(f"  vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}")

            elif key == 's':
                vx = max(-1.0, vx - VX_STEP)
                send_current_twist()
                print(f"  vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}")

            elif key == 'a':
                vy = min(0.5, vy + VY_STEP)
                send_current_twist()
                print(f"  vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}")

            elif key == 'd':
                vy = max(-0.5, vy - VY_STEP)
                send_current_twist()
                print(f"  vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}")

            elif key == 'q':
                wz = min(1.5, wz + WZ_STEP)
                send_current_twist()
                print(f"  vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}")

            elif key == 'e':
                wz = max(-1.5, wz - WZ_STEP)
                send_current_twist()
                print(f"  vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}")

            elif key == '0':
                vx, vy, wz = 0.0, 0.0, 0.0
                ctrl.stop_twist_control()
                twist_active = False
                print("  停止运动")

            elif key == '1':
                ctrl.set_gait("low")
            elif key == '2':
                ctrl.set_gait("mid")
            elif key == '3':
                ctrl.set_gait("high")

            elif key == 'i':
                print(f"\n  {ctrl.get_state_str()}\n")

    except KeyboardInterrupt:
        pass
    finally:
        ctrl.soft_estop()
        ctrl.stop()
        if sys.platform != 'win32':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\n程序退出")


if __name__ == "__main__":
    main()
