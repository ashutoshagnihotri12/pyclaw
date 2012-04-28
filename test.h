#include "common.h"

#include <stdio.h>

template<class a, class b, class c>
struct setOfThree
{
	a first;
	b second;
	c third;
};

struct functor1
{
	void operator()(int a)
	{
		for (int i = 0; i < a; i++)
			printf("%i",a);
	}
};

struct functor2
{
	void operator()(int a, int b)
	{
		printf("%d",a*b);
	}
};

struct functor3
{
	int operator()(int a, int b)
	{
		int c = 1;
		for (int i = 0; i < b; i++)
			c *=a;
		return c;
	}
};

template<class M>
int function(M funcs, int a, int b)
{
	printf("Using the set of functions:\n");
	printf("func 1:");
	funcs.first(a);
	printf("\nfunc 2:");
	funcs.second(a,b);
	printf("\nfunc 3: %d", funcs.third(a,b));
	printf("\n");

	return funcs.third(a,b);
}

/* in main

	//setOfThree<functor1, functor2, functor3> functions;

	//functor1 one;
	//functor2 two;
	//functor3 three;

	//functions.first = one;
	//functions.second = two;
	//functions.third = three;

	//int l = function(functions, 2, 3);

	//printf("\n\nhey, this is the result: %d\n",l);

	//printf("%d\n", 55);
*/