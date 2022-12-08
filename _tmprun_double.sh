for nm in 100 40 20 10 5 
do
    for nl in 60 50 40 30 20 15
    do
        for lw in  120 100 70 40 30 20
        do
            python _temprun_triple_credit.py "$nm" "$nl" "$lw"
        done
    done
done

for nm in 5 40 20 10 100
do
    for nl in 15 60 50 40 30 20
    do
        for lw in  20 100 70 40 30 120
        do
            python _temprun_double_credit.py "$nm" "$nl" "$lw"
        done
    done
done

for nm in 5 40 20 10 100
do
    for nl in 15 60 50 40 30 20
    do
        for lw in  20 100 70 40 30 120
        do
            python _temprun_single_credit.py "$nm" "$nl" "$lw"
        done
    done
done



