function [average_millage, average_carbon] = consumptEmission (concepts, carbon, millage, target_concept)
    tam = size(concepts);
    average_m = 0;
    average_c = 0;
    cont = 0;
    for i = 1: tam(2:2)
        if (strcmp (concepts(1,i), target_concept) == 1)
           average_m += millage(i);
           average_c += carbon(i);
           cont += 1;
        endif
    end
    average_millage = average_m / cont;
    average_carbon = average_c / cont;
endfunction
