#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>


int main(int argc, char *argv[]) {

    char* matrix_folder = "matrices";
    DIR* fd;
    FILE* results_csv;
    struct dirent* in_file;

    results_csv = fopen("results.csv","w");

    /* Scanning the in directory */
    if (NULL == (fd = opendir (matrix_folder))) 
    {
        fprintf(stderr, "Error : Failed to open input directory - %s\n", strerror(errno));
        return 1;
    }

    fprintf(results_csv, "product, format, threads");
    while ((in_file = readdir(fd))) 
  	{
        if (!strcmp (in_file->d_name, "."))
            continue;
        if (!strcmp (in_file->d_name, ".."))    
            continue;

        fprintf(results_csv, ", %s", in_file->d_name);
    
    }

    fprintf(results_csv, "\n");

    fclose(results_csv);

    closedir(fd);
}