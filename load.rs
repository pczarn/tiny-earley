use crate::{Rule, Symbol};

use super::Grammar;

#[macro_export]
macro_rules! grammar {
    (
        S = [$($name:pat),+]
        R = {
            $($lhs:ident ::= $($rhs:ident)+;)+
        }
    ) => {
        {
            let syms = [$(stringify!($name)),+];
            let mut grammar = Grammar::new(syms, 0);
            let [$($name),+] = grammar.symbols();
            $(
                grammar.rule(
                    $lhs,
                    [$($rhs),+]
                );
            )+
            grammar
        }
    };
}

#[test]
fn grammar() {
    let grammar = grammar! {
        S = [start, a, b, c, d]
        R = {
            start ::= a b c;
            a ::= b c d;
        }
    };
    let [start, a, b, c, d] = grammar.symbols();
    assert_eq!(grammar.symbol_names, ["start", "a", "b", "c", "d"]);
    assert_eq!(grammar.rules.into_iter().take(1).collect::<Vec<_>>(), vec![Rule { lhs: Symbol(5), rhs0: Symbol(1), rhs1: Some(Symbol(2)), id: None }]);
}

