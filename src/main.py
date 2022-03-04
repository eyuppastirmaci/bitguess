from process import PreProcess


def main():
    pre_process = PreProcess()

    # Sample
    turkish_example_word = '1941 yılında astronomlar, 100 inç (2.500 mm)lik bir teleskop kullanarak süpernovadan arta kalan gaz kalıntısını tespit ettiler. Bu süpernovadan kaynaklanan kalıntı, türünün "prototip" nesnelerinden biri olarak kabul edilir ve hâlen astronomlarca üzerinde pek çok çalışmalar yapılmaktadır.'
    print(turkish_example_word)
    print("_____________________")
    print(pre_process.extract_stop_words(turkish_example_word))
    print("_____________________")
    print(pre_process.stem_words(turkish_example_word))
    print("_____________________")
    print(pre_process.part_of_speech(turkish_example_word))
    print("_____________________")
    print(pre_process.named_entity_recognition(turkish_example_word))


if __name__ == '__main__':
    main()
