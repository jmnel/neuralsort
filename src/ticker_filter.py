import re

import settings

NYSE_TEST_TICKERS = {'ATEST', 'CTEST', 'MTEST', 'NTEST', 'ZTST'}

NASDAQ_TEST_TICKERS = {'ZAZZT', 'ZBZZT', 'ZJZZT', 'ZVZZT'}

# On NASDAQ a ticker might have a 5 letter name. The 5th letter conveys special meaning.
# A - class A shares
# B - class B shares
# C - NextShares ETMF (type of ETF)
# D - new issue
# E - used to denote dlinquency in SEC filings
# F - foreign issue
# G - first convertible bond
# H - second convertible bond
# I - third convertible bond
# J - voting; temporarily denotes shareholder vote situation
# K - non-voting
# L - miscellaneous situations; seems to be bonds and preferred stock
# M - fourth preferred issue
# N - third preferred issue
# O - second preferred issue
# P - first preferred issue
# Q - this used to be used to indicate bankruptcy
# R - rights
# S - shares of beneficial interest
# T - securities with warrants or rights
# U - units
# V - when issued or when distributed; shares that are set to split or similar corporate action
# W - warrants
# X - mutual fund quotation service
# Y - american depository receipt
# Z - micellaneous situtations
NASDAQ_POSTFIXES = {'C', 'D', 'E', 'G', 'H', 'I',
                    'J', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'W', 'V', 'W', 'X', 'Z'}

NYSE_POSTFIXES = {'F', 'Q', 'I', 'Z', 'L', 'N', 'O', 'C', 'CL', 'P',
                  'WS', 'WD', 'U', 'V', 'W', 'R', '', 'V'}

NYSE_REGEX_TERMS = [
    r'\bdeposit[ao]ry',
    r'\bbeneficial \binterest',
    r'\binterest', r'\bunits', r'\bfund', r'\bdue', r'\bbond',
    r'\betf$', r'\betn\b', r'\badr\b', r'\bdepsitary\b']
NYSE_REGEX = re.compile('|'.join(NYSE_REGEX_TERMS), flags=re.IGNORECASE)

NASDAQ_REGEX_TERMS = [
    r'\bfund\b', r'\bdeposit[ao]ry\b', r'\bproshares\b', r'\bpowershares\b', r'\biShares\b',
    r'\bvictoryshares\b', r'\bdepsitary\b',
    r'\betf\b', r'\bbond\b', r'\betn\b', r'\badr\b', r'\bet$', r'\bdue\b', r'\%']

NASDAQ_REGEX = re.compile('|'.join(NASDAQ_REGEX_TERMS), flags=re.IGNORECASE)


class TickerFilter:

    def filter(ticker, name, exchange):

        if len(ticker) < 1:
            raise ValueError('Ticker must be non-empty string')

        if exchange not in settings.EXCHANGES:
            raise ValueError(
                f'Exchange must be one of {settings.EXCHANGES}: got {exchange}')

        # Route to NYSE filter.
        if exchange in ('NYSE', 'NYSE MKT'):
            return TickerFilter.filter_nyse(ticker, name)

        # All NYSE Arca are discared; contains ETFs.
        elif exchange == 'NYSE Arca':
            return False

        # Route to NASDAQ filter.
        elif exchange == 'NASDAQ':
            return TickerFilter.filter_nasqad(ticker, name)

    def filter_nyse(ticker, name):

        # Filter out NYSE test tickers.
        if ticker in NYSE_TEST_TICKERS:
            return False

        # Check for dot convention.
        tokens = ticker.split('.')
        if len(tokens) > 1:
            root = tokens[0]

            # Check for NYSE test tickers.
            if root in NYSE_TEST_TICKERS:
                return False

            # Filter out non-common stock postfixes after 1st dot.
            if tokens[1] in NYSE_POSTFIXES:
                return False

            # Postic A or B following dot, denotes class A or class B common stock.
            elif tokens[1] in {'A', 'B'}:
                pass

                #                return True

        # Check for 5th letter ticker.
        if len(ticker) == 5:

            postfix = ticker[-1]

            # Filter by 5th letter for NYSE postfixes.
            if postfix in NYSE_POSTFIXES:
                return False

        # Some bonds will still remain with no clear way to tell from the ticker alone.
        # Use regex to search for specific patterns.
        if '%' in name:
            return False

        if NYSE_REGEX.search(name):
            return False

        # Otherwise, ticker is probably a NYSE common stock.
        return True

    def filter_nasqad(ticker, name):

        # Filter out NASDAQ test tickers.
        if ticker in NASDAQ_TEST_TICKERS:
            return False

        # Check if ticker has NASDAQ 5th letter.
        if len(ticker) == 5:

            # Check 5th letter of ticker.

            # For some reason GOOGL breaks from convention.
            if ticker == 'GOOGL':
                return True

            postfix = ticker[-1]
            if postfix in NASDAQ_POSTFIXES:
                return False

        # There might still be some categories that are not reflected in ticker, so parse name with
        # regex expression looking for certain patterns and keywords.
        if NASDAQ_REGEX.search(name):
            return False

        # Otherwise, ticker is probably a NASDAQ common stock.
        return True
