import sys

if __name__ == "__main__":

    nb_upss = 6
    ups_capacity = 100
    upss_capacities = [ups_capacity] * nb_upss
    # racks_load = [33] * nb_upss
    racks_load = [66] * nb_upss
    ups_loads = [100] * nb_upss

    # Their algorithm
    i = 0
    while i < nb_upss:
        if i + 2 < nb_upss:
            if racks_load[i] + racks_load[i+1] + racks_load[i+2] <= upss_capacities[i+1]:
                ups_loads[i + 1] = racks_load[i] + racks_load[i+1] + racks_load[i+2]
                ups_loads[i] = 0
                ups_loads[i+2] = 0
                i += 3
            else:
                if i + 3 < nb_upss:
                    if racks_load[i+1] + racks_load[i+2] + racks_load[i+3] <= upss_capacities[i+2]:
                        ups_loads[i + 2] = racks_load[i+1] + racks_load[i+2] + racks_load[i+3]
                        ups_loads[i+1] = 0
                        ups_loads[i+3] = 0
                        i += 4
                    else:
                        if racks_load[i] + racks_load[i+1] <= upss_capacities[i]:
                            ups_loads[i] = racks_load[i] + racks_load[i+1]
                            ups_loads[i+1] = 0
                            i += 2
                        else:
                            i += 1
                else:
                    if i + 1 < nb_upss:
                        if racks_load[i] + racks_load[i+1] <= upss_capacities[i]:
                            ups_loads[i] = racks_load[i] + racks_load[i+1]
                            ups_loads[i+1] = 0
                            i += 2
                        else:
                            i += 1
                    else:
                        break
        else:
            break

    # Print result
    print(ups_loads)

    sys.exit(0)
