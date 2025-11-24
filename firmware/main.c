#include "ch32fun.h"
#include <stdint.h>

#define OUTPUT_MASK ((uint16_t)((1u << 0) | (1u << 1) | (1u << 2) | (1u << 3)))

int main(void)
{
	SystemInit();

	/* Enable GPIOC clock (and AFIO to keep remap sane) */
	RCC->APB2PCENR |= RCC_APB2Periph_AFIO | RCC_APB2Periph_GPIOC;

	/* Configure PC0-PC3 as 10MHz push-pull outputs */
	uint32_t cfg = GPIOC->CFGLR;
	cfg &= ~(0xFFFFu);      /* clear config for pins 0-3 */
	cfg |= 0x1111u;         /* MODE=01 (10MHz) | CNF=00 for pins 0-3 */
	GPIOC->CFGLR = cfg;

	while (1) {
		for (uint8_t active = 0; active < 4; active++) {
			/* Reset all PC0-PC3, then set the active one high */
			GPIOC->BSHR = ((uint32_t)OUTPUT_MASK << 16) | (1u << active);
			Delay_Ms(1); /* 1 ms per step */
		}
	}
}
