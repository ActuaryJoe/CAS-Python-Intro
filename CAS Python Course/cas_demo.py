
import re
from pathlib import Path
from IPython.display import display, Markdown

class Demo():
    def __init__(self, fn, debug_mode=False):
        """
        Class Demo() Run a Demo

        Maintains a position in a document and lets you move
        forward or backwards. Shows a table of contents.

        Functions

        toc(levels)     show table of contents; highlight current location
        code()          insert code from current section
        s()             s[how] the current section
        show(n)         show section n; look at toc for numbering
        show_all()      show the entire document
        f()             move forward to next section
        b()             move backwards to previous section
        object(n)       same as show(n)

        """

        self.fn = fn
        self.p = Path(fn)
        if self.p.exists() == False:
            self.p = Path(__file__).parent / self.p
        self.txt = self.p.open('r').read()
        self.stxt = re.split('^###? ', self.txt, flags=re.MULTILINE)
        self.section = 0
        self.max_sections = len(self.stxt) - 1
        self._ip = get_ipython()
        self._toc = re.findall('^(#[#]*) ([^\n]+)\n', self.txt, flags=re.MULTILINE)
        self.debug_mode = debug_mode

    def toc(self, levels=1):
        """
        return table of contents to levels levels
        """
        levels+=2
        ans = []
        for i, l in enumerate(self._toc):
            depth = len(l[0])
            if depth < levels:
                depth -= 2
                pad = '### ' if len(l[0])==1 else  f'{i}. '
                if depth == 0 and i == self.section:
                    ans.append(f'{pad} ***{l[1]}***')
                elif depth == 1 and i == self.section:
                    ans.append(f'{pad} ***{l[1]}***')
                elif depth == 0:
                    ans.append(f'{pad} **{l[1]}**')
                else:
                    ans.append(f'{pad} {l[1]}')
        ans. append(f'\nAt position {self.section} of {self.max_sections-1}; '
                f'  {100*self.section/(self.max_sections-1):.0f}% complete.')
        display(Markdown('\n'.join(ans)))

    def code(self):
        """
        ni = next input
        if there is code in txt put in next input
        """
        txt = self.stxt[self.section]
        stxt = txt.split('\n')
        in_code = False
        block = []
        for l in stxt:
            if l == '```python':
                in_code = True
            elif l == '```':
                if in_code:
                    in_code = False
                else:
                    raise ValueError('end block with  no start block')
            elif in_code:
                block.append(l)
        if len(block):
            code = '\n'.join(block)
            self._ip.set_next_input(code) # , replace=True)

    def show(self, section):
        if section >= 0 and section < self.max_sections:
            self.section = section
            self.s()
        if section < 0:
            self.show(self.section + section)

    def show_all(self):
        display(Markdown(self.txt))

    def s(self):
        # reload
        if self.debug_mode:
            ss = self.section
            self.__init__(self.fn)
            self.section = ss

        t = self.stxt[self.section]
        st = t.split('\n')
        if st[0].lower().find('solution')==0:
            # want to put the question
            # find the question
            lt = self.stxt[self.section-1].split('Exercise')
            if len(lt) > 1:
                st.insert(1, lt[1])
            t = '\n'.join(st)
        if self.section:
            t = '## ' + t
        display(Markdown(t))
        self.code()

    def f(self):
        """ forwards """
        if self.section < self.max_sections:
            self.section += 1
        self.s()

    def b(self):
        """ backwards """
        if self.section > 0:
            self.section -= 1
        self.s()

    def __call__(self, section=None):
        if section is None:
            self.f()
        elif section == 0:
            # repeat current section
            self.show(self.section)
        else:
            self.show(section)

    # def __repr__(self):
    #     self.f()
    #     return ''
