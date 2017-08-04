from foxai import adder_ai


def main():
    adder = adder_ai.AdderAI()
    adder.train()

    print("Manual test")
    print("===========")
    print("Which number do you want to add together?")
    while True:
        a = int(input('a: '))
        b = int(input('b: '))
        result = adder.add(a, b)
        print("Y = %s" % result)

if __name__ == "__main__":
    main()
