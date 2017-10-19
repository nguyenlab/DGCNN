#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 
/* DEFINE NODE*/
struct var
    {
        char key[20];
        char value[200];
        struct var *next;
    };
void print_var(var* _var)
{
	printf("(k,v) =(%s, %s)\n", _var->key, _var->value);
}   
void print_list(var* list)
{
	printf("\nvar list:\n");
	var* p= list;
	while(p!=NULL)
	{
		print_var(p);
		p = p->next;
	}
} 
var* NewVar(char key[20], char value[200])
{
    var *newvar = (var*)malloc(sizeof(var));
    snprintf( newvar->key, 20, "%s",key);
    snprintf( newvar->value, 200, "%s",value);                                 

    newvar->next = NULL;                                             
     
    return(newvar);                                             
}
void Insert(var **head, var *newvar)
{   
    newvar->next = *head;
    *head = newvar;
} 
char* getvarvalue(var* list, const char key[20])
{
	var* p = list;
	char * value = NULL;
	while(p!= NULL)
	{
		if (strcmp(p->key, key) == 0)
		{
			value = p->value;
			break;
		}
		p = p->next;
	}
	return value;
}  
char ** str_split(char* str, char* delims, int maxitems)
{
	char ** results = (char**)malloc(maxitems*sizeof(char*));
    char *pt;
    pt = strtok (str,delims);
    int count =0;
    while (pt != NULL&& count< maxitems) 
	{
		int len = strlen(pt)+1;
		results[count] = (char*)malloc(len*sizeof(char));
		snprintf( results[count], len, "%s",pt);
        pt = strtok (NULL, delims);
        count++;
    }
    return results;
}
var * getvarfromstring(char*varinfor, char* delims)
{
	char** kv = str_split(varinfor, delims,2);
	return NewVar(kv[0], kv[1]);
}
//int main()
//{
//	char str1[] ="k1:v1";
//	char str2[] ="k2:v2";
//	char str3[] ="k3:v3";
////    char ** test = str_split(str, ":",2);
////    if(test[1]==NULL)
////        printf("NULL");
////    printf("'%s', '%s'", test[0], test[1]);
//    
//	var* v1 = getvarfromstring(str1);
//	var* v2 = getvarfromstring(str2);
//	var* v3 = getvarfromstring(str3);
//	var** v=(var**)malloc(sizeof(var*));
//	*v = NULL;
//	Insert(v, v1);
//	Insert(v, v2);
//	Insert(v, v3);
//	print_list(*v);
//	char *value1 = getvarvalue(*v,"k1");
//	if (value1 != NULL)
//		printf("%s", value1);
//
//}    
