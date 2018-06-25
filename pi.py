from random import random


def main():
    c = 10000000
    n = 0

    for _ in range(c):
        # ランダムな点を作ります。
        x = random() * 2 - 1
        y = random() * 2 - 1

        # 円の内側かどうかを判定します。
        if x ** 2 + y ** 2 < 1:
            n += 1

    print(4 * n / c)  # 私が試したときは、3.1410432になりました。


if __name__ == '__main__':
    main()
