for nm in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120
do
    for nl in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120
    do
        for lw in  5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120
        do
            for arc in "single" "double" "triple"
            do

                python _print.py "$nm" "$nl" "$lw" "$arc"
                pkill python
                pkill python3
            done
        done
    done
done