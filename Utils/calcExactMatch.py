
def calc_exact_match(references, predictions):
    refs = [x.strip() for x in open(references, 'r', encoding='utf-8').readlines()]
    pres = [x.strip() for x in open(predictions, 'r', encoding='utf-8').readlines()]
    
    assert len(refs) == len(pres)

    exact = 0
    total = 0

    length = len(refs)
    count = 0
    for i in range(length):
      r = refs[i]
      p = pres[i]
      if r.strip() == p.strip():
        exact +=1
      total += 1
      print(r)
      print(p)
      print("_"*10)

    exact_match = str(exact * 100.0/total)
    print("Exact Match: ", str(exact * 100.0/total))
    return exact_match
