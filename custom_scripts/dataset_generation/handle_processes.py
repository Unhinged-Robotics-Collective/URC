import multiprocessing as mp
from typing import Protocol, Any, Iterable


def test_mp():
    def foo(qi: mp.Queue, qo: mp.Queue):
        while True:
            o = qi.get()
            if o == "AAA":
                return
            qo.put(o)


    if __name__ == '__main__':
        mp.set_start_method('spawn')
        qi = mp.Queue()
        qo = mp.Queue()
        p = mp.Process(target=foo, args=(qi, qo))
        p.start()
        while True:
            i = input()
            qi.put(i)
            o = qo.get()
            print(o)
            if o == "q":
                qi.put("AAA")
                print("QUIT")
                break
        p.join()


def test_handler():
    def test_io(*, q_i: mp.Queue, q_o: mp.Queue, **kwargs):
        i = q_i.get()
        o = i + "AAAAAAA"
        q_o.put(o)

    q1 = mp.Queue()
    q2 = mp.Queue()
    p = get_handler(test_io, q_i=q1, q_o=q2)
    p.start()
    q1.put("lel")
    print(q2.get())
    p.join()



class SingleFn(Protocol):
    def __call__(self, **kwargs: Any) -> None: ...


def get_handler(single_f: SingleFn, **kwargs) -> mp.Process:
    p = mp.Process(target=single_f, kwargs=kwargs)
    return p

def activate_all(ps: Iterable[mp.Process], keyword: str):
    list(map(lambda x: exec("x." + keyword + "()"), ps))


if __name__ == "__main__":
    test_handler()
