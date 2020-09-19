function [color, concept] = lettering (indice)
  if (indice == 1)
    color = 'g*';
    concept = 'Otimo';
    media = 0;
     elseif (indice == 2)
       color = 'b*';
       concept = 'Bom';
     elseif (indice == 3)
       color = 'c*';
       concept = 'Regular';
     elseif (indice == 4)
       color = 'y*';
       concept = 'Ruim';
     else
       color = 'r*';
       concept = 'Pessimo';
  endif
endfunction
